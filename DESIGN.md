# AgriShield Frontend Design

## World

Regional-atlas operating surface on an unbleached cream ground. Flat green, coral, yellow, blue, and lilac map lobes carry regional identity and risk state. Ink text remains the primary contrast; borders are hairlines rather than glow decoration.

## First View

The first viewport contains the active regional forecast, a geographic lobe field, and a narrow March-2026 readout. Navigation stays in a compact pill bar. The primary action leads to evidence; the secondary action opens scenario analysis.

## Components

- DotField supplies a restrained spatial background.
- ScrambledText adds regional-name character motion.
- RotatingText changes the forecast signal phrase.
- TextType provides typed analyst framing.
- GooeyNav owns the three primary views.
- PromptBar carries analyst briefing requests.
- LatticeLoader represents loading.
- GlowCursor provides a low-intensity local pointer field.

## Interaction

Region selection updates every chart and alert surface. API health is always visible. API prediction failures fall back to the static browser model. Custom-region provenance remains visible.

## Responsive Behavior

Desktop uses a three-column atlas readout. Tablet compresses the readout below the map. Mobile uses a single-column atlas and horizontal heatmap scroll.
