"""
Maps Sonic Pi sample symbols to absolute file paths.

Sonic Pi built-in samples live in  <sonic_pi_root>/etc/samples/
and are referenced in code as  :bd_haus  :ambi_choir  etc.

Usage:
    resolver = SampleResolver("/path/to/sonic-pi")
    path = resolver.resolve("bd_haus")   # or resolve(":bd_haus")
    path = resolver.resolve(":bd_haus", rate=1.0)   # kwargs ignored here
"""
from __future__ import annotations
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Hardcoded list of all built-in sample names (derived from etc/samples/)
# ---------------------------------------------------------------------------

BUILTIN_SAMPLES: list[str] = [
    # ambi
    "ambi_choir", "ambi_dark_woosh", "ambi_drone", "ambi_glass_hum",
    "ambi_glass_rub", "ambi_haunted_hum", "ambi_lunar_land", "ambi_piano",
    "ambi_sauna", "ambi_soft_buzz", "ambi_swoosh",
    # arovane
    "arovane_beat_a", "arovane_beat_b", "arovane_beat_c",
    "arovane_beat_d", "arovane_beat_e",
    # bass
    "bass_dnb_f", "bass_drop_c", "bass_hard_c", "bass_hit_c",
    "bass_thick_c", "bass_trance_c", "bass_voxy_c", "bass_voxy_hit_c",
    "bass_woodsy_c",
    # bd (bass drum)
    "bd_808", "bd_ada", "bd_boom", "bd_chip", "bd_fat", "bd_gas",
    "bd_haus", "bd_jazz", "bd_klub", "bd_mehackit", "bd_pure",
    "bd_sone", "bd_tek", "bd_zome", "bd_zum",
    # drum
    "drum_bass_hard", "drum_bass_soft", "drum_cowbell",
    "drum_cymbal_closed", "drum_cymbal_hard", "drum_cymbal_open",
    "drum_cymbal_pedal", "drum_cymbal_soft", "drum_heavy_kick",
    "drum_roll", "drum_snare_hard", "drum_snare_soft",
    "drum_splash_hard", "drum_splash_soft",
    "drum_tom_hi_hard", "drum_tom_hi_soft",
    "drum_tom_lo_hard", "drum_tom_lo_soft",
    "drum_tom_mid_hard", "drum_tom_mid_soft",
    # elec
    "elec_beep", "elec_bell", "elec_blip", "elec_blip2", "elec_blup",
    "elec_bong", "elec_chime", "elec_cymbal", "elec_filt_snare",
    "elec_flip", "elec_fuzz_tom", "elec_hi_snare", "elec_hollow_kick",
    "elec_lo_snare", "elec_mid_snare", "elec_ping", "elec_plip",
    "elec_pop", "elec_snare", "elec_soft_kick", "elec_tick",
    "elec_triangle", "elec_twang", "elec_twip", "elec_wood",
    # glitch
    "glitch_bass_g", "glitch_perc1", "glitch_perc2", "glitch_perc3",
    "glitch_perc4", "glitch_perc5", "glitch_robot1", "glitch_robot2",
    # guit
    "guit_e_fifths", "guit_e_slide", "guit_em9", "guit_harmonics",
    # hat
    "hat_bdu", "hat_cab", "hat_cymbal", "hat_edu", "hat_gem",
    "hat_liquid", "hat_metal", "hat_pseudo", "hat_raw", "hat_snap",
    "hat_snap2",
    # loop
    "loop_4x4", "loop_amen", "loop_amen_full", "loop_breakbeat",
    "loop_compus", "loop_garzul", "loop_industrial", "loop_mika",
    "loop_safari", "loop_tabla",
    # mehackit
    "mehackit_robot1", "mehackit_robot2", "mehackit_robot3",
    "mehackit_robot4", "mehackit_robot5", "mehackit_robot6",
    "mehackit_robot7", "mehackit_robot8",
    # misc
    "misc_burp", "misc_cineboom", "misc_crow",
    # perc
    "perc_bell", "perc_bell2", "perc_dry", "perc_snap",
    "perc_snap2", "perc_swash", "perc_till",
    # sn (snare)
    "sn_dof", "sn_dub", "sn_generic", "sn_zome",
    # tabla
    "tabla_bas_a", "tabla_bas_b", "tabla_bas_c", "tabla_bas_d",
    "tabla_bas_e", "tabla_bas_f", "tabla_bayan_a", "tabla_bayan_b",
    "tabla_bayan_c", "tabla_bayan_d", "tabla_bayan_e", "tabla_bayan_f",
    "tabla_bayan_ghe", "tabla_bayan_mute", "tabla_bayan_open",
    "tabla_dayan_a", "tabla_dayan_b", "tabla_dayan_c", "tabla_dayan_d",
    "tabla_dayan_e", "tabla_dayan_f", "tabla_dayan_g",
    "tabla_ghe1", "tabla_ghe2", "tabla_ghe3", "tabla_ghe4",
    "tabla_ghe5", "tabla_ghe6", "tabla_ghe7", "tabla_ghe8",
    "tabla_ke1", "tabla_ke2", "tabla_ke3",
    "tabla_na1", "tabla_na2", "tabla_na3", "tabla_na4",
    "tabla_na5", "tabla_na6", "tabla_na7",
    "tabla_re1", "tabla_re2", "tabla_re3", "tabla_re4",
    "tabla_te1", "tabla_te2", "tabla_te3", "tabla_te_ne1",
    "tabla_te_ne2", "tabla_te_ne3",
    "tabla_tin1", "tabla_tin2", "tabla_tin3",
    "tabla_tu1", "tabla_tu2", "tabla_tu3",
    # vinyl
    "vinyl_backspin", "vinyl_hiss", "vinyl_rewind", "vinyl_scratch",
]


class SampleResolver:
    """
    Resolves Sonic Pi sample symbols / strings to absolute file paths.

    Args:
        sonic_pi_root: Absolute path to the sonic-pi repository root.
                       If None, only string paths are resolved.
    """

    def __init__(self, sonic_pi_root: str | None = None):
        self._root = Path(sonic_pi_root) if sonic_pi_root else None
        self._cache: dict[str, str] = {}

        if self._root:
            self._samples_dir = self._root / "etc" / "samples"
            self._build_cache()

    def _build_cache(self):
        """Scan the samples directory and populate name → path mapping."""
        if not self._samples_dir.exists():
            return
        for p in self._samples_dir.iterdir():
            if p.suffix.lower() in ('.flac', '.wav', '.aif', '.aiff', '.ogg', '.mp3'):
                # stem becomes the key (e.g. "bd_haus")
                self._cache[p.stem.lower()] = str(p)

    def resolve(self, name: object) -> str | None:
        """
        Resolve a sample name to a file path.

        Accepts:
          - Symbol string without ':' :  'bd_haus'
          - Symbol string with ':'    : ':bd_haus'
          - Absolute file path string : '/path/to/sample.wav'
        Returns None if the sample cannot be found.
        """
        if not isinstance(name, str):
            return None

        name = name.lstrip(':').strip()

        # Absolute / relative path passed directly
        if os.sep in name or '/' in name:
            return name if os.path.exists(name) else None

        # Lookup in cache
        key = name.lower()
        if key in self._cache:
            return self._cache[key]

        # Fallback: construct expected path even if file doesn't exist yet
        # (useful when synthdefs dir is known but samples haven't been scanned)
        if self._root:
            for ext in ('.flac', '.wav', '.aif', '.aiff', '.ogg', '.mp3'):
                p = self._samples_dir / f"{name}{ext}"
                if p.exists():
                    return str(p)
        return None

    def is_builtin(self, name: str) -> bool:
        """Return True if the name is a known built-in sample."""
        return name.lstrip(':').lower() in BUILTIN_SAMPLES

    def all_builtin_names(self) -> list[str]:
        return list(BUILTIN_SAMPLES)


# ---------------------------------------------------------------------------
# Module-level convenience (works without a sonic-pi root for known names)
# ---------------------------------------------------------------------------

_default_resolver: SampleResolver | None = None


def init(sonic_pi_root: str):
    """Initialise the module-level resolver with the sonic-pi root path."""
    global _default_resolver
    _default_resolver = SampleResolver(sonic_pi_root)


def resolve(name: object) -> str | None:
    """Resolve using the module-level resolver (call init() first)."""
    if _default_resolver is None:
        raise RuntimeError("Call sample_map.init(sonic_pi_root) before using resolve()")
    return _default_resolver.resolve(name)
