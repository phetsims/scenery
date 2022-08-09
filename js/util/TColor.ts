// Copyright 2021-2022, University of Colorado Boulder

import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import { Color } from '../imports.js';

/**
 * Type representing a ColorDef
 * Please see Color.toColor() for a way to transform these colors.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TColor = TReadOnlyProperty<Color | string | null> | TReadOnlyProperty<Color | string> | TReadOnlyProperty<Color> | TReadOnlyProperty<string> | Color | string | null;

export default TColor;