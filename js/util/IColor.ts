// Copyright 2021, University of Colorado Boulder

import IReadOnlyProperty from '../../../axon/js/IReadOnlyProperty.js';
import { Color } from '../imports.js';

/**
 * Type representing a ColorDef
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type IColor = IReadOnlyProperty<Color | string | null> | IReadOnlyProperty<Color | string> | IReadOnlyProperty<Color> | Color | string | null;

export default IColor;