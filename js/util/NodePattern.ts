// Copyright 2013-2022, University of Colorado Boulder

/**
 * Creates a pattern based on a Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import { Pattern, scenery, Node } from '../imports.js';

class NodePattern extends Pattern {
  constructor( node: Node, resolution: number, x: number, y: number, width: number, height: number ) {
    assert && assert( resolution > 0 && Number.isInteger( resolution ), 'Resolution should be a positive integer' );
    assert && assert( Number.isInteger( width ) );
    assert && assert( Number.isInteger( height ) );

    const imageElement = document.createElement( 'img' );

    // NOTE: This callback is executed SYNCHRONOUSLY
    function callback( canvas: HTMLCanvasElement, x: number, y: number, width: number, height: number ) {
      imageElement.src = canvas.toDataURL();
    }
    const tmpNode = new Node( {
      scale: resolution,
      children: [ node ]
    } );
    tmpNode.toCanvas( callback, -x * resolution, -y * resolution, width * resolution, height * resolution );

    super( imageElement );

    this.setTransformMatrix( Matrix3.scaling( 1 / resolution ) );
  }
}

scenery.register( 'NodePattern', NodePattern );
export default NodePattern;