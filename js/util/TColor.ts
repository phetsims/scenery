// Copyright 2021-2026, University of Colorado Boulder

/**
 * Type representing a ColorDef
 * Please see Color.toColor() for a way to transform these colors.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import { TReadOnlyProperty } from '../../../axon/js/TReadOnlyProperty.js';
import Color from '../util/Color.js';

type TColor = TReadOnlyProperty<Color | string | null> |
  TReadOnlyProperty<Color | string> |
  TReadOnlyProperty<Color | null> |
  TReadOnlyProperty<string | null> |
  TReadOnlyProperty<Color> |
  TReadOnlyProperty<string> |
  TReadOnlyProperty<null> |
  Color | string | null;

export default TColor;