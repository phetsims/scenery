// Copyright 2022, University of Colorado Boulder

/**
 * Creates a pattern based on a Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import { Node, Pattern, scenery } from '../imports.js';

export default class NodePattern extends Pattern {
  public constructor( node: Node, resolution: number, x: number, y: number, width: number, height: number, matrix = Matrix3.IDENTITY ) {
    assert && assert( resolution > 0 && Number.isInteger( resolution ), 'Resolution should be a positive integer' );
    assert && assert( Number.isInteger( width ) );
    assert && assert( Number.isInteger( height ) );

    const imageElement = document.createElement( 'img' );

    // NOTE: This callback is executed SYNCHRONOUSLY
    function callback( canvas: HTMLCanvasElement, x: number, y: number, width: number, height: number ): void {
      imageElement.src = canvas.toDataURL();
    }
    const tmpNode = new Node( {
      scale: resolution,
      children: [ node ]
    } );
    tmpNode.toCanvas( callback, -x * resolution, -y * resolution, width * resolution, height * resolution );

    super( imageElement );

    this.setTransformMatrix( matrix.timesMatrix( Matrix3.scaling( 1 / resolution ) ) );
  }
}

scenery.register( 'NodePattern', NodePattern );
