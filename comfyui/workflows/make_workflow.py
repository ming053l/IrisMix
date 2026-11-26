"""Write the example IrisMix workflow for ComfyUI in both formats:
  irismix_workflow.json      UI format (drag onto the ComfyUI canvas)
  irismix_workflow_api.json  API format (POST /prompt)
The images are the repository examples (assets/examples); copy them into ComfyUI/input first (see comfyui/README.md).
usage: python make_workflow.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = [('source-warm.webp', 'Target photo'), ('reference-cobalt.webp', 'Eyes reference'),
          ('reference-merlot.webp', 'Lips reference'), ('reference-poppy.webp', 'Face paint reference')]
EDIT = dict(strength=1.0, eyes=True, lips=True, face=True, forehead=True, lip_color='#b0303a', eyes_hue=0, face_hue=-120, forehead_hue=0)

nodes, links = [], []


def node(type_, pos, size, widgets=(), inputs=(), outputs=(), title=None):
    n = dict(id=len(nodes) + 1, type=type_, pos=list(pos), size=list(size), flags={}, order=len(nodes), mode=0,
             inputs=[dict(name=a, type=t, link=None) for a, t in inputs],
             outputs=[dict(name=a, type=t, links=[], slot_index=k) for k, (a, t) in enumerate(outputs)],
             properties={'Node name for S&R': type_}, widgets_values=list(widgets))
    if title: n['title'] = title
    nodes.append(n); return n


def link(a, ao, b, bi):
    i = len(links) + 1; a['outputs'][ao]['links'].append(i); b['inputs'][bi]['link'] = i
    links.append([i, a['id'], ao, b['id'], bi, a['outputs'][ao]['type']])


LAYER = 'IRISMIX_LAYER'
dec = []
for k, (f, t) in enumerate(PHOTOS):
    y = k * 330
    ld = node('LoadImage', (0, y), (250, 310), [f, 'image'], outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], title=t)
    dc = node('IrisMixDecompose', (280, y + 40), (250, 80), inputs=[('photo', 'IMAGE')],
              outputs=[('bare_face', 'IMAGE'), ('makeup_layer', 'IMAGE'), ('layer', LAYER)],
              title='Remove target makeup' if k == 0 else 'Decompose reference')
    pv = node('PreviewImage', (560, y), (240, 310), inputs=[('images', 'IMAGE')], title='Bare face' if k == 0 else 'Makeup layer')
    link(ld, 0, dc, 0); link(dc, 0 if k == 0 else 1, pv, 0); dec.append(dc)
cp = node('IrisMixCompose', (850, 300), (300, 150), inputs=[('target', 'IMAGE'), ('eyes_from', LAYER), ('lips_from', LAYER),
          ('face_from', LAYER), ('forehead_from', LAYER)], outputs=[('composite', 'IMAGE'), ('look', 'IRISMIX_LOOK')], title='Compose by part')
link(dec[0], 0, cp, 0)
for k, d in enumerate(dec[1:]): link(d, 2, cp, k + 1)
pc = node('PreviewImage', (850, 490), (440, 600), inputs=[('images', 'IMAGE')], title='Composite (warp only)')
link(cp, 0, pc, 0)
rd = node('IrisMixRender', (1320, 300), (300, 90), [0, 'fixed'], inputs=[('look', 'IRISMIX_LOOK')],
          outputs=[('rendered', 'IMAGE'), ('result', 'IRISMIX_RESULT')], title='Render (one step)')
link(cp, 1, rd, 0)
pr = node('SaveImage', (1320, 490), (440, 600), ['irismix/rendered'], inputs=[('images', 'IMAGE')], title='Rendered')
link(rd, 0, pr, 0)
ed = node('IrisMixEditLayer', (1820, 120), (440, 340), list(EDIT.values()), inputs=[('result', 'IRISMIX_RESULT')],
          outputs=[('edited', 'IMAGE')], title='Edit layer (no new render)')
link(rd, 1, ed, 0)
pe = node('SaveImage', (1820, 490), (440, 600), ['irismix/edited'], inputs=[('images', 'IMAGE')], title='Edited')
link(ed, 0, pe, 0)
groups = [dict(title='Photos', bounding=[-20, -60, 290, 1390], color='#3f789e'),
          dict(title='Decompose', bounding=[265, -60, 545, 1390], color='#88aa88'),
          dict(title='Compose and render', bounding=[835, 140, 940, 970], color='#b06634'),
          dict(title='Edit the layer', bounding=[1805, 40, 470, 1070], color='#a1309b')]
ui = dict(last_node_id=len(nodes), last_link_id=len(links), nodes=nodes, links=links, groups=groups, config={},
          extra=dict(ds=dict(scale=0.75, offset=[60, 90])), version=0.4)
json.dump(ui, open(f'{HERE}/irismix_workflow.json', 'w'), indent=1)

# API format: the same graph, widget values by input name
WIDGETS = {'LoadImage': ['image'], 'IrisMixRender': ['seed'], 'SaveImage': ['filename_prefix'], 'IrisMixEditLayer': list(EDIT)}
api = {}
for n in nodes:
    inp = dict(zip(WIDGETS.get(n['type'], []), n['widgets_values']))
    for i in n['inputs']:
        if i['link'] is not None:
            l = next(x for x in links if x[0] == i['link']); inp[i['name']] = [str(l[1]), l[2]]
    api[str(n['id'])] = {'class_type': n['type'], 'inputs': inp, '_meta': {'title': n.get('title', n['type'])}}
json.dump(api, open(f'{HERE}/irismix_workflow_api.json', 'w'), indent=1)
print(len(nodes), 'nodes', len(links), 'links')
