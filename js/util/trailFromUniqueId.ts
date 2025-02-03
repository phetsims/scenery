// Copyright 2025, University of Colorado Boulder

/**
 * Re-create a trail to a root node from an existing Trail id. The rootNode must have the same Id as the first
 * Node id of uniqueId.
 *
 * @author Jesse Greenberg
 */


import { PDOM_UNIQUE_ID_SEPARATOR } from '../accessibility/pdom/PDOM_UNIQUE_ID_SEPARATOR.js';
import type Node from '../nodes/Node.js';
import Trail from './Trail.js';

/**
 * @param rootNode - the root of the trail being created
 * @param uniqueId - integers separated by ID_SEPARATOR, see getUniqueId
 */
export const trailFromUniqueId = ( rootNode: Node, uniqueId: string ): Trail => {
  const trailIds = uniqueId.split( PDOM_UNIQUE_ID_SEPARATOR );
  const trailIdNumbers = trailIds.map( id => Number( id ) );

  let currentNode = rootNode;

  const rootId = trailIdNumbers.shift();
  const nodes = [ currentNode ];

  assert && assert( rootId === rootNode.id );

  while ( trailIdNumbers.length > 0 ) {
    const trailId = trailIdNumbers.shift();

    // if accessible order is set, the trail might not match the hierarchy of children - search through nodes
    // in pdomOrder first because pdomOrder is an override for scene graph structure
    const pdomOrder = currentNode.pdomOrder || [];
    const children = pdomOrder.concat( currentNode.children );
    for ( let j = 0; j < children.length; j++ ) {

      // pdomOrder supports null entries to fill in with default order
      if ( children[ j ] !== null && children[ j ]!.id === trailId ) {
        const childAlongTrail = children[ j ]!;
        nodes.push( childAlongTrail );
        currentNode = childAlongTrail;

        break;
      }

      assert && assert( j !== children.length - 1, 'unable to find node from unique Trail id' );
    }
  }

  return new Trail( nodes );
};