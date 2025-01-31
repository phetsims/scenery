// Copyright 2025, University of Colorado Boulder

/**
 * This constant is set up to allow us to change our unique id strategy. Both strategies have trade-offs that are
 * described in https://github.com/phetsims/phet-io/issues/1847#issuecomment-1068377336. TRAIL_ID is our path forward
 * currently, but will break PhET-iO playback if any Nodes are created in the recorded sim OR playback sim but not
 * both. Further information in the above issue and https://github.com/phetsims/phet-io/issues/1851.
 *
 * @author Jesse Greenberg
 */

import PDOMUniqueIdStrategy from './PDOMUniqueIdStrategy.js';

const UNIQUE_ID_STRATEGY = PDOMUniqueIdStrategy.TRAIL_ID;
export default UNIQUE_ID_STRATEGY;