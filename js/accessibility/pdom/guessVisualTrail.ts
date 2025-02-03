// Copyright 2025, University of Colorado Boulder

/**
 * Since a "Trail" on PDOMInstance can have discontinuous jumps (due to pdomOrder), this finds the best
 * actual visual Trail to use, from the trail of a PDOMInstance to the root of a Display.
 *
 * @param trail - trail of the PDOMInstance, which can containe "gaps"
 * @param rootNode - root of a Display
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

import type Node from '../../nodes/Node.js';
import type Trail from '../../util/Trail.js';

export const guessVisualTrail = ( trail: Trail, rootNode: Node ): Trail => {
  trail.reindex();

  // Search for places in the trail where adjacent nodes do NOT have a parent-child relationship, i.e.
  // !nodes[ n ].hasChild( nodes[ n + 1 ] ).
  // NOTE: This index points to the parent where this is the case, because the indices in the trail are such that:
  // trail.nodes[ n ].children[ trail.indices[ n ] ] = trail.nodes[ n + 1 ]
  const lastBadIndex = trail.indices.lastIndexOf( -1 );

  // If we have no bad indices, just return our trail immediately.
  if ( lastBadIndex < 0 ) {
    return trail;
  }

  const firstGoodIndex = lastBadIndex + 1;
  const firstGoodNode = trail.nodes[ firstGoodIndex ];
  const baseTrails = firstGoodNode.getTrailsTo( rootNode );

  // firstGoodNode might not be attached to a Display either! Maybe client just hasn't gotten to it yet, so we
  // fail gracefully-ish?
  // assert && assert( baseTrails.length > 0, '"good node" in trail with gap not attached to root')
  if ( baseTrails.length === 0 ) {
    return trail;
  }

  // Add the rest of the trail back in
  const baseTrail = baseTrails[ 0 ];
  for ( let i = firstGoodIndex + 1; i < trail.length; i++ ) {
    baseTrail.addDescendant( trail.nodes[ i ] );
  }

  assert && assert( baseTrail.isValid(), `trail not valid: ${trail.uniqueId}` );

  return baseTrail;
};