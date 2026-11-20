# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Sri Lanka dengue early-warning: reusable project code (imported by notebooks and the app)."""
import os

# joblib/loky tries to count *physical* CPU cores with the Windows `wmic` tool, which newer
# Windows 11 builds no longer ship, and prints a noisy (harmless) traceback. Setting this
# variable below the logical core count skips that probe; it also leaves one core free so
# the laptop stays responsive while models train with n_jobs=-1.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 2) - 1, 1)))
