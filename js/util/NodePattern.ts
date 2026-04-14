// Copyright 2022-2026, University of Colorado Boulder

/**
 * Creates a pattern based on a Node.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Node from '../nodes/Node.js';
import Pattern from '../util/Pattern.js';
import scenery from '../scenery.js';
import { scratchContext } from './scratches.js';

export default class NodePattern extends Pattern {
  public constructor( node: Node, resolution: number, x: number, y: number, width: number, height: number, matrix = Matrix3.IDENTITY ) {
    assert && assert( resolution > 0 && Number.isInteger( resolution ), 'Resolution should be a positive integer' );
    assert && assert( Number.isInteger( width ) );
    assert && assert( Number.isInteger( height ) );

    const imageElement = document.createElement( 'img' );
    let renderedCanvas: HTMLCanvasElement | null = null;

    // NOTE: This callback is executed SYNCHRONOUSLY
    function callback( canvas: HTMLCanvasElement, x: number, y: number, width: number, height: number ): void {
      renderedCanvas = canvas;
      imageElement.src = canvas.toDataURL();
    }

    const tmpNode = new Node( {
      scale: resolution,
      children: [ node ]
    } );
    tmpNode.toCanvas( callback, -x * resolution, -y * resolution, width * resolution, height * resolution );

    super( imageElement );

    // Use the already-rendered canvas tile for CanvasPattern creation so canvas-only rasterization paths
    // (like screenshot export) do not depend on asynchronous image decoding of the data URL.
    assert && assert( renderedCanvas, 'NodePattern tile canvas should have been created synchronously' );
    imageElement.width = width * resolution;
    imageElement.height = height * resolution;
    this.canvasPattern = scratchContext.createPattern( renderedCanvas!, 'repeat' )!;

    this.setTransformMatrix( matrix.timesMatrix( Matrix3.scaling( 1 / resolution ) ) );
  }
}

scenery.register( 'NodePattern', NodePattern );
