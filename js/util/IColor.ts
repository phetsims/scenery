// Copyright 2021, University of Colorado Boulder

import IProperty from '../../../axon/js/IProperty.js';
import { Color } from '../imports.js';

/**
 * Type representing a ColorDef
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type IColor = IProperty<Color | string | null> | IProperty<Color | string> | IProperty<Color> | Color | string | null;

export default IColor;