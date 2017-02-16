// Copyright 2013-2017, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var MultiListener = require( 'SCENERY/listeners/MultiListener' );

  /**
   * @constructor
   *
   * @params {Object} [options] - See the constructor body (below) for documented options.
   */
  function PanZoomListener( targetNode, panBounds, options ) {
    options = _.extend( {
      allowScale: true,
      allowRotation: false,
      pressCursor: null
    }, options );

    MultiListener.call( this, targetNode, options );

    this._panBounds = panBounds;
  }

  scenery.register( 'PanZoomListener', PanZoomListener );

  inherit( MultiListener, PanZoomListener, {
    reposition: function() {
      MultiListener.prototype.reposition.call( this );

      // Assume same scale in each dimension
      var currentScale = this._targetNode.getScaleVector().x;
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
  } );

  return PanZoomListener;
} );
