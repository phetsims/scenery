// Copyright 2021, University of Colorado Boulder

import IProperty from '../../../axon/js/IProperty.js';
import { Paint, Color } from '../imports.js';

/**
 * Type representing a PaintDef
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type IPaint = Paint | IProperty<Color | string | null> | IProperty<Color | string> | IProperty<Color> | Color | string | null;

export default IPaint;