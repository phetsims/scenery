// Copyright 2021-2022, University of Colorado Boulder

import IReadOnlyProperty from '../../../axon/js/IReadOnlyProperty.js';
import { Color } from '../imports.js';

/**
 * Type representing a ColorDef
 * Please see Color.toColor() for a way to transform these colors.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type IColor = IReadOnlyProperty<Color | string | null> | IReadOnlyProperty<Color | string> | IReadOnlyProperty<Color> | IReadOnlyProperty<string> | Color | string | null;

export default IColor;