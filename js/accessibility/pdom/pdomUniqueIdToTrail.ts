// Copyright 2025, University of Colorado Boulder

/**
 * @author Jesse Greenberg
 */

import UNIQUE_ID_STRATEGY from './UNIQUE_ID_STRATEGY.js';
import type Display from '../../display/Display.js';
import type Trail from '../../util/Trail.js';
import PDOMUniqueIdStrategy from './PDOMUniqueIdStrategy.js';
import { trailFromUniqueId } from '../../util/trailFromUniqueId.js';

/**
 * @param display
 * @param uniqueId - value returned from PDOMInstance.getPDOMInstanceUniqueId()
 * @returns null if there is no path to the unique id provided.
 */
export const pdomUniqueIdToTrail = ( display: Display, uniqueId: string ): Trail | null => {
  if ( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.INDICES ) {
    return display.getTrailFromPDOMIndicesString( uniqueId );
  }
  else {
    assert && assert( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.TRAIL_ID );
    return trailFromUniqueId( display.rootNode, uniqueId );
  }
};