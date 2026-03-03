"""

HOW TO USE

1.  Run training:
        python pacman_rl.py --mode train
2.  Watch the trained agent play:
        python pacman_rl.py --mode play
3.  Continue training from a checkpoint:
        python pacman_rl.py --mode train --model pacman_dqn.pth

Requires:  torch  numpy  pygame  tcod
    pip install torch numpy pygame tcod

"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import collections
import math
import os
import random
import time
from copy import deepcopy
from typing import Tuple, List, Optional

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Import from game file.

from pacman_final import (
    PacmanGameController,
    GameRenderer,
    Hero,
    Ghost,
    Wall,
    Cookie,
    Direction,
    translate_maze_to_screen,
    translate_screen_to_maze,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
UNIFIED_SIZE   = 32
ACTION_MAP     = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
    4: Direction.NONE,   # STAY
}
N_ACTIONS = len(ACTION_MAP)

# Feature-vector state dimensions
# [ dist_nearest_pellet, dx_pellet, dy_pellet,           (3)
#   dist_ghost1..4, dx_ghost1..4, dy_ghost1..4,          (4 × 3 = 12)
#   current_dir_onehot (5),                               (5)
#   wall_up, wall_down, wall_left, wall_right             (4) ]
STATE_DIM = 3 + 12 + 5 + 4   # = 24

# Collision threshold – hero & ghost overlap when centres are within this
COLLISION_DIST = UNIFIED_SIZE * 0.8

# ─────────────────────────────────────────────────────────────────────────────
# Gym-style Environment Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class PacmanEnv:
    """
    Wraps the existing pygame Pacman game into a gym-style interface.

    reset()        → state (np.ndarray)
    step(action)   → state, reward, done, info
    render()       → (call pygame display)
    get_state()    → state (np.ndarray)

    Headless mode (render=False) suppresses the pygame window by using a
    hidden display surface so all the existing draw() calls still work
    without crashing, but nothing is shown on screen.
    """

    # Reward constants
    R_PELLET        =  1.0
    R_DIE           = -20.0
    R_WIN           = 50.0
    R_STEP          = -0.01   # small time penalty
    R_REVERSE       = -0.5    # penalty for bouncing back and forth

    MAX_STEPS_PER_EPISODE = 2000

    def __init__(self, render: bool = False):
        self._do_render = render
        self._controller: Optional[PacmanGameController] = None
        self._renderer:   Optional[GameRenderer]         = None
        self._hero:       Optional[Hero]                 = None
        self._ghosts:     List[Ghost]                    = []
        self._step_count  = 0
        self._prev_cookie_count = 0
        self._total_cookies     = 0
        self._score             = 0
        self._alive             = True
        self._won               = False
        self._prev_action       = 4  # STAY
        self._prev_prev_action  = 4
        self._done = False

        if not pygame.get_init():
            pygame.init()

        if not render:
            # Headless: use an off-screen surface
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Tear down any existing game and start a fresh episode."""
        self._teardown()
        self._build_game()
        self._step_count = 0
        self._score = 0
        self._alive = True
        self._won = False
        self._done = False
        self._total_cookies = len(self._renderer.get_cookies())
        self._prev_cookie_count = self._total_cookies
        # Pump one tick so positions are initialised
        self._pump_pygame()
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action, advance the game one logical frame, return SARS tuple.

        action: int  0=UP 1=DOWN 2=LEFT 3=RIGHT 4=STAY
        """
        assert not self._done, "Call reset() before stepping after done."

        # ── Apply action ──────────────────────────────────────────────────
        if self._hero is not None:
            self._hero.set_direction(ACTION_MAP[action])

        # ── Advance game logic (no display flip) ──────────────────────────
        self._tick_game_objects()
        self._pump_pygame()

        # ── Check ghost collisions (base game doesn't do this) ────────────
        died = self._check_ghost_collision()

        # ── Count remaining cookies (game removes them from game_objects) ─
        current_cookies = len([
            c for c in self._renderer.get_cookies()
            if c in self._renderer.get_game_objects()
        ])
        cookies_eaten = self._prev_cookie_count - current_cookies
        self._score += cookies_eaten * 10

        # ── Compute reward ─────────────────────────────────────────────────
        reward = self.R_STEP
        if cookies_eaten > 0:
            reward += self.R_PELLET * cookies_eaten
        if died:
            reward += self.R_DIE
            self._alive = False

        # ── Check terminal conditions ──────────────────────────────────────
        self._step_count += 1
        won = (current_cookies == 0)
        timeout = self._step_count >= self.MAX_STEPS_PER_EPISODE

        if won:
            reward += self.R_WIN
            self._won = True

        self._done = won or died or timeout

        # Reverse direction penalty: UP<->DOWN or LEFT<->RIGHT
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if opposite.get(self._prev_action) == action and \
           opposite.get(self._prev_prev_action) == self._prev_action:
            reward += self.R_REVERSE

        # Track for next step
        self._prev_prev_action = self._prev_action
        self._prev_action      = action
        self._prev_cookie_count = current_cookies

        state = self.get_state()
        info  = {
            "score":   self._score,
            "won":     won,
            "timeout": timeout,
        }
        return state, reward, self._done, info

    def render(self):
        """Flip the pygame display (only meaningful in render=True mode)."""
        if self._do_render:
            pygame.display.flip()

    def get_state(self) -> np.ndarray:
        """
        Feature-vector state.

        Returns a float32 array of shape (STATE_DIM,) = (24,).

        Features (all normalised to ~[-1, 1] or [0, 1]):
          [0]    distance to nearest pellet  (/ max_dist)
          [1,2]  dx, dy to nearest pellet    (/ max_dist)
          [3..14] for each of up to 4 ghosts: dist, dx, dy  (/ max_dist)
          [15..19] one-hot current direction  [UP,DOWN,LEFT,RIGHT,NONE]
          [20..23] wall in each direction     {0,1}
        """
        if self._hero is None or not self._alive:
            return np.zeros(STATE_DIM, dtype=np.float32)

        W = self._renderer._width
        H = self._renderer._height
        max_dist = math.sqrt(W * W + H * H)

        hx, hy = self._hero.get_position()

        # ── Nearest pellet (only those still in game_objects) ─────────────
        game_objs = self._renderer.get_game_objects()
        cookies = [c for c in self._renderer.get_cookies() if c in game_objs]
        if cookies:
            dists = [math.hypot(c.x - hx, c.y - hy) for c in cookies]
            idx   = int(np.argmin(dists))
            near  = cookies[idx]
            p_dist = dists[idx] / max_dist
            p_dx   = (near.x - hx) / max_dist
            p_dy   = (near.y - hy) / max_dist
        else:
            p_dist, p_dx, p_dy = 0.0, 0.0, 0.0

        pellet_feats = [p_dist, p_dx, p_dy]

        # ── Up to 4 ghosts ────────────────────────────────────────────────
        ghost_feats = []
        for i in range(4):
            if i < len(self._ghosts):
                g = self._ghosts[i]
                d  = math.hypot(g.x - hx, g.y - hy) / max_dist
                dx = (g.x - hx) / max_dist
                dy = (g.y - hy) / max_dist
                ghost_feats += [d, dx, dy]
            else:
                ghost_feats += [1.0, 0.0, 0.0]   # ghost absent → far away

        # ── Direction one-hot ─────────────────────────────────────────────
        dir_order = [Direction.UP, Direction.DOWN, Direction.LEFT,
                     Direction.RIGHT, Direction.NONE]
        cur_dir   = self._hero.current_direction if self._hero else Direction.NONE
        dir_oh    = [1.0 if cur_dir == d else 0.0 for d in dir_order]

        # ── Walls in each direction ───────────────────────────────────────
        walls = []
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            collides, _ = self._hero.check_collision_in_direction(d)
            walls.append(1.0 if collides else 0.0)

        state = np.array(
            pellet_feats + ghost_feats + dir_oh + walls,
            dtype=np.float32,
        )
        return state

    def close(self):
        self._teardown()
        pygame.quit()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_game(self):
        """Reconstruct the entire game from scratch."""
        self._controller = PacmanGameController()
        size = self._controller.size

        W = size[0] * UNIFIED_SIZE
        H = size[1] * UNIFIED_SIZE

        if self._do_render:
            self._renderer = GameRenderer(W, H)
        else:
            # Build renderer but its display surface won't be shown
            self._renderer = _HeadlessRenderer(W, H)

        # Walls
        for y, row in enumerate(self._controller.numpy_maze):
            for x, col in enumerate(row):
                if col == 0:
                    self._renderer.add_wall(Wall(self._renderer, x, y, UNIFIED_SIZE))

        # Cookies
        for cs in self._controller.cookie_spaces:
            t = translate_maze_to_screen(cs)
            self._renderer.add_cookie(
                Cookie(self._renderer, t[0] + UNIFIED_SIZE / 2, t[1] + UNIFIED_SIZE / 2)
            )

        # Ghosts
        self._ghosts = []
        for i, gs in enumerate(self._controller.ghost_spawns):
            t = translate_maze_to_screen(gs)
            ghost = Ghost(
                self._renderer, t[0], t[1], UNIFIED_SIZE,
                self._controller,
                self._controller.ghost_colors[i % 4],
            )
            self._renderer.add_game_object(ghost)
            self._ghosts.append(ghost)

        # Hero
        self._hero = Hero(self._renderer, UNIFIED_SIZE, UNIFIED_SIZE, UNIFIED_SIZE)
        self._renderer.add_hero(self._hero)

    def _teardown(self):
        self._renderer   = None
        self._controller = None
        self._hero       = None
        self._ghosts     = []

    def _tick_game_objects(self):
        """Run one logical tick for every game object (no drawing)."""
        if self._renderer is None:
            return
        for obj in list(self._renderer.get_game_objects()):
            obj.tick()

    def _pump_pygame(self):
        """
        Process pygame events without blocking on user input.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._done = True

    def _check_ghost_collision(self) -> bool:
        """Return True if the hero overlaps with any ghost."""
        if self._hero is None:
            return False
        hx, hy = self._hero.get_position()
        for g in self._ghosts:
            dist = math.hypot(g.x - hx, g.y - hy)
            if dist < COLLISION_DIST:
                return True
        return False

    def _compute_reward(self, action: int) -> float:
        """Legacy helper – reward is now computed inline in step()."""
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Headless renderer subclass
# ─────────────────────────────────────────────────────────────────────────────
class _HeadlessRenderer(GameRenderer):
    """
    Same as GameRenderer but uses a hidden surface and never calls
    pygame.display.flip(), so the game loop can run without a window.
    """
    def __init__(self, in_width: int, in_height: int):
        pygame.init()
        self._width   = in_width
        self._height  = in_height
        self._screen  = pygame.Surface((in_width, in_height))  # off-screen
        self._clock   = pygame.time.Clock()
        self._done    = False
        self._game_objects = []
        self._walls   = []
        self._cookies = []
        self._hero    = None


# ─────────────────────────────────────────────────────────────────────────────
# DQN Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    """
    Simple 3-layer MLP for the feature-vector state.
    Input:  STATE_DIM = 24
    Output: N_ACTIONS = 5
    """
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Standard DQN with:
      • experience replay
      • target network (hard update every C steps)
      • ε-greedy exploration with linear decay
    """

    def __init__(
        self,
        state_dim:      int   = STATE_DIM,
        n_actions:      int   = N_ACTIONS,
        lr:             float = 1e-4,
        gamma:          float = 0.99,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.05,
        epsilon_decay:  int   = 100_000,  # steps over which ε decays
        batch_size:     int   = 64,
        replay_capacity:int   = 50_000,
        target_update:  int   = 1_000,    # hard update every N steps
        device:         str   = "auto",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps_done    = 0

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(replay_capacity)
        self.loss_fn   = nn.SmoothL1Loss()  # Huber loss

    # ── epsilon ───────────────────────────────────────────────────────────────
    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-self.steps_done / self.epsilon_decay)

    # ── action selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Pure exploitation – used at play time."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    # ── store & learn ─────────────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.replay.push(
            np.array(state, dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )
        self.steps_done += 1

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states      = torch.tensor(np.array(batch.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action,               dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward,               dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done,                 dtype=torch.float32, device=self.device)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a')  (zero if terminal)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1).values
            targets    = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Hard-update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str = "pacman_dqn.pth"):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer":         self.optimizer.state_dict(),
            "steps_done":        self.steps_done,
        }, path)
        print(f"  ✓ Model saved → {path}")

    def load(self, path: str = "pacman_dqn.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)
        print(f"  ✓ Model loaded ← {path}  (step {self.steps_done})")


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train(
    n_episodes:     int  = 3_000,
    save_path:      str  = "pacman_dqn.pth",
    load_path:      Optional[str] = None,
    print_every:    int  = 50,
    save_every:     int  = 200,
):
    """
    Train the DQN agent in headless mode.

    Arguments
    ---------
    n_episodes   – total episodes to run
    save_path    – where to save the model
    load_path    – (optional) .pth file to continue training from
    print_every  – print progress every N episodes
    save_every   – save checkpoint every N episodes
    """
    env   = PacmanEnv(render=False)
    agent = DQNAgent()

    if load_path and os.path.exists(load_path):
        agent.load(load_path)

    rewards_window = collections.deque(maxlen=100)
    total_steps    = 0

    print(f"Training on device: {agent.device}")
    print(f"State dim: {STATE_DIM}  |  Actions: {N_ACTIONS}")
    print("─" * 60)

    for ep in range(1, n_episodes + 1):
        state    = env.reset()
        ep_reward = 0.0
        done     = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.learn()
            state      = next_state
            ep_reward += reward
            total_steps += 1

        rewards_window.append(ep_reward)

        if ep % print_every == 0:
            avg = sum(rewards_window) / len(rewards_window)
            print(
                f"Ep {ep:5d}/{n_episodes} | "
                f"reward: {ep_reward:8.2f} | "
                f"avg100: {avg:8.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"score: {info['score']:5d} | "
                f"steps: {total_steps:,}"
            )

        if ep % save_every == 0:
            agent.save(save_path)

    agent.save(save_path)
    env.close()
    print("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Play with Trained Model
# ─────────────────────────────────────────────────────────────────────────────
def play_with_model(
    model_path: str   = "pacman_dqn.pth",
    fps:        int   = 30,
    n_games:    int   = 5,
):
    """
    Load a saved model and watch it play in a pygame window.

    Controls
    --------
    ESC / close window  → quit
    """
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Train first.")
        return

    agent = DQNAgent()
    agent.load(model_path)
    agent.policy_net.eval()

    env = PacmanEnv(render=True)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 20)
    big_font = pygame.font.SysFont("monospace", 60)

    for game_idx in range(1, n_games + 1):
        state  = env.reset()
        done   = False
        total_reward = 0.0

        while not done:
            # ── Render ────────────────────────────────────────────────────
            env._renderer._screen.fill((0, 0, 0))
            for obj in env._renderer.get_game_objects():
                obj.draw()

            # Score overlay
            score_surf = font.render(
                f"[Score: {env._score}]", True, (255, 255, 255)
            )
            env._renderer._screen.blit(score_surf, (5, 5))

            if not env._alive:
                msg = big_font.render("YOU DIED", True, (255, 0, 0))
                rect = msg.get_rect(center=(env._renderer._width // 2,
                                            env._renderer._height // 2))
                env._renderer._screen.blit(msg, rect)

            if env._won:
                msg = big_font.render("YOU WON", True, (0, 255, 0))
                rect = msg.get_rect(center=(env._renderer._width // 2,
                                            env._renderer._height // 2))
                env._renderer._screen.blit(msg, rect)

            pygame.display.flip()

            # ── Events ────────────────────────────────────────────────────
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True
            if quit_requested:
                env.close()
                return

            # ── Agent step ────────────────────────────────────────────────
            action = agent.select_action_greedy(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            clock.tick(fps)

        print(
            f"Game {game_idx}/{n_games} | "
            f"Score: {info['score']} | "
            f"Won: {info['won']} | "
            f"Reward: {total_reward:.2f}"
        )

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pacman DQN RL Agent")
    parser.add_argument("--mode",    choices=["train", "play"], default="train")
    parser.add_argument("--model",   default="pacman_dqn.pth",  help="Path to .pth file")
    parser.add_argument("--episodes",type=int, default=3_000,   help="Training episodes")
    parser.add_argument("--fps",     type=int, default=30,      help="FPS for play mode")
    parser.add_argument("--games",   type=int, default=5,       help="Games to play in play mode")
    args = parser.parse_args()

    if args.mode == "train":
        train(
            n_episodes=args.episodes,
            save_path=args.model,
            load_path=args.model if os.path.exists(args.model) else None,
        )
    else:
        play_with_model(model_path=args.model, fps=args.fps, n_games=args.games)