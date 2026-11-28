// The ReactBits components are vendored JSX without types; this is the one place they are cast.
import type { ComponentType } from 'react'
import DotField from './DotField.jsx'
import GlowCursor from './GlowCursor.jsx'
import GooeyNav from './GooeyNav.jsx'
import LatticeLoader from './LatticeLoader.jsx'
import PromptBar from './PromptBar.jsx'
import RotatingText from './RotatingText.jsx'
import ScrambledText from './ScrambledText.jsx'
import TextType from './TextType.jsx'

type Untyped = ComponentType<Record<string, unknown>>

export const DotFieldView = DotField as unknown as Untyped
export const GlowCursorView = GlowCursor as unknown as Untyped
export const GooeyNavView = GooeyNav as unknown as Untyped
export const LatticeLoaderView = LatticeLoader as unknown as Untyped
export const PromptBarView = PromptBar as unknown as Untyped
export const RotatingTextView = RotatingText as unknown as Untyped
export const ScrambledTextView = ScrambledText as unknown as Untyped
export const TextTypeView = TextType as unknown as Untyped
