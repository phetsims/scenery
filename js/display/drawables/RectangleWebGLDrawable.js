// Copyright 2016-2022, University of Colorado Boulder

/**
 * WebGL drawable for Rectangle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import ReadOnlyProperty from '../../../../axon/js/ReadOnlyProperty.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { Color, RectangleStatefulDrawable, Renderer, scenery, WebGLSelfDrawable } from '../../imports.js';

const scratchColor = new Color( 'transparent' );

class RectangleWebGLDrawable extends RectangleStatefulDrawable( WebGLSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    if ( !this.vertexArray ) {
      // format [X Y R G B A] for all vertices
      this.vertexArray = new Float32Array( 6 * 6 ); // 6-length components for 6 vertices (2 tris).
    }

    // corner vertices in the relative transform root coordinate space
    this.upperLeft = new Vector2( 0, 0 );
    this.lowerLeft = new Vector2( 0, 0 );
    this.upperRight = new Vector2( 0, 0 );
    this.lowerRight = new Vector2( 0, 0 );

    this.transformDirty = true;
    this.includeVertices = true; // used by the processor
  }

  /**
   * @public
   *
   * @param {WebGLBlock} webglBlock
   */
  onAddToBlock( webglBlock ) {
    this.webglBlock = webglBlock; // TODO: do we need this reference?
    this.markDirty();
  }

  /**
   * @public
   *
   * @param {WebGLBlock} webglBlock
   */
  onRemoveFromBlock( webglBlock ) {
  }

  /**
   * @public
   * @override
   */
  markTransformDirty() {
    this.transformDirty = true;

    super.markTransformDirty();
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   * @override
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
    if ( !super.update() ) {
      return false;
    }

    if ( this.dirtyFill ) {
      this.includeVertices = this.node.hasFill();

      if ( this.includeVertices ) {
        const fill = ( this.node.fill instanceof ReadOnlyProperty ) ? this.node.fill.value : this.node.fill;
        const color = scratchColor.set( fill );
        const red = color.red / 255;
        const green = color.green / 255;
        const blue = color.blue / 255;
        const alpha = color.alpha;

        for ( let i = 0; i < 6; i++ ) {
          const offset = i * 6;
          this.vertexArray[ 2 + offset ] = red;
          this.vertexArray[ 3 + offset ] = green;
          this.vertexArray[ 4 + offset ] = blue;
          this.vertexArray[ 5 + offset ] = alpha;
        }
      }
    }

    if ( this.transformDirty || this.dirtyX || this.dirtyY || this.dirtyWidth || this.dirtyHeight ) {
      this.transformDirty = false;

      const x = this.node._rectX;
      const y = this.node._rectY;
      const width = this.node._rectWidth;
      const height = this.node._rectHeight;

      const transformMatrix = this.instance.relativeTransform.matrix; // with compute need, should always be accurate
      transformMatrix.multiplyVector2( this.upperLeft.setXY( x, y ) );
      transformMatrix.multiplyVector2( this.lowerLeft.setXY( x, y + height ) );
      transformMatrix.multiplyVector2( this.upperRight.setXY( x + width, y ) );
      transformMatrix.multiplyVector2( this.lowerRight.setXY( x + width, y + height ) );

      // first triangle XYs
      this.vertexArray[ 0 ] = this.upperLeft.x;
      this.vertexArray[ 1 ] = this.upperLeft.y;
      this.vertexArray[ 6 ] = this.lowerLeft.x;
      this.vertexArray[ 7 ] = this.lowerLeft.y;
      this.vertexArray[ 12 ] = this.upperRight.x;
      this.vertexArray[ 13 ] = this.upperRight.y;

      // second triangle XYs
      this.vertexArray[ 18 ] = this.upperRight.x;
      this.vertexArray[ 19 ] = this.upperRight.y;
      this.vertexArray[ 24 ] = this.lowerLeft.x;
      this.vertexArray[ 25 ] = this.lowerLeft.y;
      this.vertexArray[ 30 ] = this.lowerRight.x;
      this.vertexArray[ 31 ] = this.lowerRight.y;
    }

    this.setToCleanState();
    this.cleanPaintableState();

    return true;
  }
}

RectangleWebGLDrawable.prototype.webglRenderer = Renderer.webglVertexColorPolygons;

scenery.register( 'RectangleWebGLDrawable', RectangleWebGLDrawable );

Poolable.mixInto( RectangleWebGLDrawable );

export default RectangleWebGLDrawable;