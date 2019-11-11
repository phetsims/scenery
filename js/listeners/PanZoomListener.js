// Copyright 2017-2019, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: unit tests
 *
 * TODO: add example usage
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( require => {
  'use strict';

  const merge = require( 'PHET_CORE/merge' );
  const MultiListener = require( 'SCENERY/listeners/MultiListener' );
  const scenery = require( 'SCENERY/scenery' );

  class PanZoomListener extends MultiListener {

    /**
     * @constructor
     * @extends MultiListener
     *
     * TODO: Have 'content' bounds (instead of using the targetNode's bounds), since some things may extend off the side
     *       of the content bounds.
     *
     * TODO: Support mutable target or pan bounds (adjust transform).
     *
     * TODO: If scale !~=1, allow interrupting other pointers when multitouch begins (say pan area is filled with button)
     *
     * @param {Node} targetNode - The Node that should be transformed by this PanZoomListener.
     * @param {Bounds2} panBounds - Bounds that should be fully filled with content at all times.
     * @param {Object} [options] - See the constructor body (below) for documented options.
     */
    constructor( targetNode, panBounds, options ) {

      options = merge( {
        allowScale: true,
        allowRotation: false,
        pressCursor: null
      }, options );

      // TODO: type checks for options

      super( targetNode, options );

      this._panBounds = panBounds;
    }

    reposition() {
      super.reposition();

      // Assume same scale in each dimension
      const currentScale = this._targetNode.getScaleVector().x;
      if ( currentScale < 1 ) {
        this._targetNode.scale( 1 / currentScale );
      }

      // Don't let panning go through
      if ( this._targetNode.left > this._panBounds.left ) {
        this._targetNode.left = this._panBounds.left;
      }
      if ( this._targetNode.top > this._panBounds.top ) {
        this._targetNode.top = this._panBounds.top;
      }
      if ( this._targetNode.right < this._panBounds.right ) {
        this._targetNode.right = this._panBounds.right;
      }
      if ( this._targetNode.bottom < this._panBounds.bottom ) {
        this._targetNode.bottom = this._panBounds.bottom;
      }
    }
  }

  return scenery.register( 'PanZoomListener', PanZoomListener );
} );
