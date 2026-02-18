// Copyright 2020-2026, University of Colorado Boulder

/**
 * SVG namespace, used for document.createElementNS( svgns, name );
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import scenery from '../scenery.js';

const svgns = 'http://www.w3.org/2000/svg';

scenery.register( 'svgns', svgns );
export default svgns;