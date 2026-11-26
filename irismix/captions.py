"""Structured per-part captions for the renderer.

Template (one clause per part, fixed part order, no identity / age / ethnicity / hair / clothes / lighting / background):
    <Part>: <colour>; <finish>; <shape and placement>; <edges> edges; <opacity>.
  Part      Eyes | Brows | Lips | Cheeks | Face | Forehead | Accents
  colour    1-3 colour names (base first)
  finish    matte | satin | glossy | metallic | shimmer | glitter
  shape     short motif phrase + landmark-relative placement
  edges     crisp | soft | diffused                                   (measured from the alpha profile)
  opacity   sheer | medium | opaque                                   (measured: median alpha of the design)
At inference the fields are measured on the decomposed reference layer (render.layer_caption) and worded by render()."""

PART_ORDER = ('Eyes', 'Brows', 'Lips', 'Cheeks', 'Face', 'Forehead', 'Accents')


def _colour_phrase(cols, part):
    if not cols: return ''
    if len(cols) == 1: return cols[0]
    return f'{cols[0]} and {cols[1]}' if len(cols) == 2 else f'{cols[0]}, {cols[1]} and {cols[2]}'


def caption_text(cap):
    """the caption text of a look: one clause per part in PART_ORDER."""
    segs = []
    for part in PART_ORDER:
        if part not in cap: continue
        c = cap[part]; f = []
        if c['colour']: f.append(_colour_phrase(c['colour'], part))
        if c['finish']: f.append(c['finish'])
        sp = '; '.join(x for x in (c['motif'], c['placement']) if x)
        if sp: f.append(sp)
        f.append(f"{c['edges']} edges"); f.append(c['opacity'])
        segs.append(f'{part}: ' + '; '.join(f) + '.')
    return ' '.join(segs)
