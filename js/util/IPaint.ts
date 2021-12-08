// Copyright 2021, University of Colorado Boulder

import IReadOnlyProperty from '../../../axon/js/IReadOnlyProperty.js';
import { Color, Paint } from '../imports.js';

/**
 * Type representing a PaintDef
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type IPaint = Paint | IReadOnlyProperty<Color | string | null> | IReadOnlyProperty<Color | string> | IReadOnlyProperty<Color> | Color | string | null;

export default IPaint;