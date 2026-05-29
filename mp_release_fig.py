import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')

# ── Standardized labels (drop r/c prefixes per manuscript convention) ──────
label_map = {
    'rLDPE/LLDPE': 'LDPE/LLDPE',
    'cPE/PP':      'PE/PP',
    'rABS':        'ABS',
    'rPET':        'PET',
    'EVA':         'EVA',
    'SBS':         'SBS',
}

# ── Visual encoding ────────────────────────────────────────────────────────
# Color = mixing process
process_colors = {'w': '#4C8EDA', 'd': '#E05C5C', 'm': '#57A86B'}
process_labels = {'w': 'Wet (w)', 'd': 'Dry (d)', 'm': 'Hybrid (m)'}

# Marker = polymer type
type_markers = {
    'LDPE/LLDPE': 'o',
    'PE/PP':      '^',
    'ABS':        's',
    'PET':        'v',
    'EVA':        'D',
    'SBS':        'X',
}

# Key series to highlight (your main finding)
highlight_series = {
    ('LDPE/LLDPE', 'm'),   # lowest MP release, logarithmic trend
    ('PET',        'd'),   # lowest MP release, logarithmic trend
}

# ── Benchmark values (read from data or set manually) ─────────────────────
eva_data = df[df['Plastic Content'].str.strip() == 'EVA']['MPs (g/m2)']
sbs_data = df[df['Plastic Content'].str.strip() == 'SBS']['MPs (g/m2)']

eva_val = eva_data.mean() if not eva_data.empty else 0.70
sbs_val = sbs_data.mean() if not sbs_data.empty else 0.34

band_lo = min(eva_val, sbs_val)
band_hi = max(eva_val, sbs_val)

# ── Build figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5))

# Shaded commercial PMB benchmark band
ax.axhspan(band_lo, band_hi, color='#E8E8E8', alpha=0.55, zorder=0,
           label='_nolegend_')
ax.axhline(band_hi, color='#999999', linewidth=0.9, linestyle='--', zorder=1)
ax.axhline(band_lo, color='#999999', linewidth=0.9, linestyle='--', zorder=1)

# Annotate benchmark lines directly on plot
x_annot = df['wt %/mass asphalt mix'].max() * 1.01
ax.text(x_annot, band_hi + 0.012, 'EVA benchmark', fontsize=7.5,
        color='#666666', va='bottom', ha='left')
ax.text(x_annot, band_lo - 0.012, 'SBS benchmark', fontsize=7.5,
        color='#666666', va='top', ha='left')

# Plot all data points
for _, row in df.iterrows():
    x     = row['wt %/mass asphalt mix']
    y     = row['MPs (g/m2)']
    method = str(row['Method']).strip()
    ptype  = label_map.get(str(row['Plastic Content']).strip(),
                           str(row['Plastic Content']).strip())

    color  = process_colors.get(method, '#333333')
    marker = type_markers.get(ptype, 'o')
    is_key = (ptype, method) in highlight_series

    if is_key:
        # Highlighted series — larger, opaque, with bold edge
        ax.scatter(x, y, color=color, marker=marker,
                   s=110, alpha=1.0, linewidths=1.8,
                   edgecolors='black', zorder=4)
    else:
        # Background series — smaller, semi-transparent
        ax.scatter(x, y, color=color, marker=marker,
                   s=55, alpha=0.45, linewidths=0.5,
                   edgecolors=color, zorder=3)

# ── Legend ─────────────────────────────────────────────────────────────────
legend_elements = []

# Group 1: Mixing process (color)
legend_elements.append(
    Line2D([0], [0], linestyle='none', label='── Mixing Process ──',
           color='none'))
for method, color in process_colors.items():
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label=process_labels[method],
               markerfacecolor=color, markersize=8))

# Group 2: Polymer type (marker)
legend_elements.append(
    Line2D([0], [0], linestyle='none', label='── Polymer Type ──',
           color='none'))
for ptype, marker in type_markers.items():
    legend_elements.append(
        Line2D([0], [0], marker=marker, color='#444444', label=ptype,
               linestyle='None', markersize=8))

# Group 3: Benchmark and highlight notes
legend_elements.append(
    Line2D([0], [0], linestyle='none', label='── Reference ──',
           color='none'))
legend_elements.append(
    Patch(facecolor='#E8E8E8', edgecolor='#999999',
          label='Commercial PMB range\n(SBS–EVA)'))
legend_elements.append(
    Line2D([0], [0], marker='o', color='w', label='Key finding\n(bold outline)',
           markerfacecolor='#888888', markeredgecolor='black',
           markeredgewidth=1.8, markersize=9))

ax.legend(handles=legend_elements,
          bbox_to_anchor=(1.02, 1), loc='upper left',
          fontsize=8, frameon=True, framealpha=0.9,
          borderpad=0.8, labelspacing=0.5)

# ── Axes formatting ────────────────────────────────────────────────────────
ax.set_xlabel('Plastic Content (wt% of total asphalt mix)', fontsize=10)
ax.set_ylabel('MP Release (g/m²)', fontsize=10)
ax.set_title(
    'Combined MP Release by Plastic Content, Polymer Type, and Mixing Process\n'
    'Enfrin et al. (2022) and Boom et al. (2023)',
    fontsize=10.5, pad=10)

# Extend x-axis slightly so rightmost points are not flush against edge
x_max = df['wt %/mass asphalt mix'].max()
ax.set_xlim(left=-0.05, right=x_max * 1.18)
ax.set_ylim(bottom=0)

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.show()

# ── Save ───────────────────────────────────────────────────────────────────
save = input("Save fig? ")
if save.lower() == 'yes':
    print("Saving fig...")
    fig.savefig("/Users/JasL/Downloads/mp_release_fig_FINAL.png",
                dpi=300, bbox_inches='tight')
    print("Fig saved!")