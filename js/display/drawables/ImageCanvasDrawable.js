// Copyright 2016, University of Colorado Boulder

/**
 * Canvas drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var ExperimentalPoolable = require( 'PHET_CORE/ExperimentalPoolable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of a Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function ImageCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }

  scenery.register( 'ImageCanvasDrawable', ImageCanvasDrawable );

  inherit( CanvasSelfDrawable, ImageCanvasDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable trait (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @public (scenery-internal)
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {ImageCanvasDrawable} - For chaining
     */
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

    /**
     * Paints this drawable to a Canvas (the wrapper contains both a Canvas reference and its drawing context).
     * @public
     *
     * Assumes that the Canvas's context is already in the proper local coordinate frame for the node, and that any
     * other required effects (opacity, clipping, etc.) have already been prepared.
     *
     * This is part of the CanvasSelfDrawable API required to be implemented for subtypes.
     *
     * @param {CanvasContextWrapper} wrapper - Contains the Canvas and its drawing context
     * @param {Node} node - Our node that is being drawn
     * @param {Matrix3} matrix - The transformation matrix applied for this node's coordinate system.
     */
    paintCanvas: function( wrapper, node, matrix ) {
      var hasImageOpacity = node._imageOpacity !== 1;

      // Ensure that the image has been loaded by checking whether it has a width or height of 0.
      // See https://github.com/phetsims/scenery/issues/536
      if ( node._image && node._image.width !== 0 && node._image.height !== 0 ) {
        // If we have image opacity, we need to apply the opacity on top of whatever globalAlpha may exist
        if ( hasImageOpacity ) {
          wrapper.context.save();
          wrapper.context.globalAlpha *= node._imageOpacity;
        }

        if ( node._mipmap && node.hasMipmaps() ) {
          var level = node.getMipmapLevel( matrix );
          var canvas = node.getMipmapCanvas( level );
          var multiplier = Math.pow( 2, level );
          wrapper.context.drawImage( canvas, 0, 0, canvas.width * multiplier, canvas.height * multiplier );
        }
        else {
          wrapper.context.drawImage( node._image, 0, 0 );
        }

        if ( hasImageOpacity ) {
          wrapper.context.restore();
        }
      }
    },

    // stateless dirty functions
    markDirtyImage: function() { this.markPaintDirty(); },
    markDirtyMipmap: function() { this.markPaintDirty(); },
    markDirtyImageOpacity: function() { this.markPaintDirty(); }
  } );

  ExperimentalPoolable.mixInto( ImageCanvasDrawable, {
    initialize: ImageCanvasDrawable.prototype.initialize
  } );

  return ImageCanvasDrawable;
} );
