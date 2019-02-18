// Copyright 2016, University of Colorado Boulder

/**
 * WebGL drawable for Rectangle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Color = require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Property = require( 'AXON/Property' );
  var RectangleStatefulDrawable = require( 'SCENERY/display/drawables/RectangleStatefulDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );

  var scratchColor = new Color( 'transparent' );

  /**
   * A generated WebGLSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * NOTE: This drawable currently only supports solid fills and no strokes.
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function RectangleWebGLDrawable( renderer, instance ) {
    this.initializeWebGLSelfDrawable( renderer, instance );

    // Stateful trait initialization
    this.initializeState( renderer, instance );

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

  scenery.register( 'RectangleWebGLDrawable', RectangleWebGLDrawable );

  inherit( WebGLSelfDrawable, RectangleWebGLDrawable, {
    webglRenderer: Renderer.webglVertexColorPolygons,

    onAddToBlock: function( webglBlock ) {
      this.webglBlock = webglBlock; // TODO: do we need this reference?
      this.markDirty();
    },

    onRemoveFromBlock: function( webglBlock ) {
    },

    // @override
    markTransformDirty: function() {
      this.transformDirty = true;

      WebGLSelfDrawable.prototype.markTransformDirty.call( this );
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !WebGLSelfDrawable.prototype.update.call( this ) ) {
        return false;
      }

      if ( this.dirtyFill ) {
        this.includeVertices = this.node.hasFill();

        if ( this.includeVertices ) {
          var fill = ( this.node.fill instanceof Property ) ? this.node.fill.value : this.node.fill;
          var color =  scratchColor.set( fill );
          var red = color.red / 255;
          var green = color.green / 255;
          var blue = color.blue / 255;
          var alpha = color.alpha;

          for ( var i = 0; i < 6; i++ ) {
            var offset = i * 6;
            this.vertexArray[ 2 + offset ] = red;
            this.vertexArray[ 3 + offset ] = green;
            this.vertexArray[ 4 + offset ] = blue;
            this.vertexArray[ 5 + offset ] = alpha;
          }
        }
      }

      if ( this.transformDirty || this.dirtyX || this.dirtyY || this.dirtyWidth || this.dirtyHeight ) {
        this.transformDirty = false;

        var x = this.node._rectX;
        var y = this.node._rectY;
        var width = this.node._rectWidth;
        var height = this.node._rectHeight;

        var transformMatrix = this.instance.relativeTransform.matrix; // with compute need, should always be accurate
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
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      // TODO: disposal of buffers?

      this.disposeState();

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    }
  } );

  RectangleStatefulDrawable.mixInto( RectangleWebGLDrawable );

  Poolable.mixInto( RectangleWebGLDrawable );

  return RectangleWebGLDrawable;
} );
