// Copyright 2021-2022, University of Colorado Boulder

import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import { Color, Paint } from '../imports.js';

/**
 * Type representing a PaintDef
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TPaint = Paint | TReadOnlyProperty<Color | string | null> | TReadOnlyProperty<Color | string> | TReadOnlyProperty<Color> | Color | string | null;

export default TPaint;