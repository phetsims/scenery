// Copyright 2020-2023, University of Colorado Boulder

/**
 * SVG namespace, used for document.createElementNS( svgns, name );
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../imports.js';

const svgns = 'http://www.w3.org/2000/svg';

scenery.register( 'svgns', svgns );
export default svgns;