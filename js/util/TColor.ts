// Copyright 2021-2024, University of Colorado Boulder

import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import { Color } from '../imports.js';

/**
 * Type representing a ColorDef
 * Please see Color.toColor() for a way to transform these colors.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TColor = TReadOnlyProperty<Color | string | null> |
  TReadOnlyProperty<Color | string> |
  TReadOnlyProperty<Color | null> |
  TReadOnlyProperty<string | null> |
  TReadOnlyProperty<Color> |
  TReadOnlyProperty<string> |
  TReadOnlyProperty<null> |
  Color | string | null;

export default TColor;