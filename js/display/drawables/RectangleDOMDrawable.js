// Copyright 2016, University of Colorado Boulder

/**
 * DOM drawable for Rectangle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var Features = require( 'SCENERY/util/Features' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var RectangleStatefulDrawable = require( 'SCENERY/display/drawables/RectangleStatefulDrawable' );
  var scenery = require( 'SCENERY/scenery' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMRectangleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory

  // scratch matrix used in DOM rendering
  var scratchMatrix = Matrix3.dirtyFromPool();

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function RectangleDOMDrawable( renderer, instance ) {
    // Super-type initialization
    this.initializeDOMSelfDrawable( renderer, instance );

    // Stateful trait initialization
    this.initializeState( renderer, instance );

    // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
    // allocation and performance costs)
    if ( !this.fillElement || !this.strokeElement ) {
      var fillElement = this.fillElement = document.createElement( 'div' );
      fillElement.style.display = 'block';
      fillElement.style.position = 'absolute';
      fillElement.style.left = '0';
      fillElement.style.top = '0';
      fillElement.style.pointerEvents = 'none';

      var strokeElement = this.strokeElement = document.createElement( 'div' );
      strokeElement.style.display = 'block';
      strokeElement.style.position = 'absolute';
      strokeElement.style.left = '0';
      strokeElement.style.top = '0';
      strokeElement.style.pointerEvents = 'none';
      fillElement.appendChild( strokeElement );
    }

    // @protected {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
    this.domElement = this.fillElement;

    // Apply CSS needed for future CSS transforms to work properly.
    scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );
  }

  scenery.register( 'RectangleDOMDrawable', RectangleDOMDrawable );

  inherit( DOMSelfDrawable, RectangleDOMDrawable, {
    /**
     * Updates our DOM element so that its appearance matches our node's representation.
     * @protected
     *
     * This implements part of the DOMSelfDrawable required API for subtypes.
     */
    updateDOM: function() {
      var node = this.node;
      var fillElement = this.fillElement;
      var strokeElement = this.strokeElement;

      if ( this.paintDirty ) {
        var borderRadius = Math.min( node._cornerXRadius, node._cornerYRadius );
        var borderRadiusDirty = this.dirtyCornerXRadius || this.dirtyCornerYRadius;

        if ( this.dirtyWidth ) {
          fillElement.style.width = node._rectWidth + 'px';
        }
        if ( this.dirtyHeight ) {
          fillElement.style.height = node._rectHeight + 'px';
        }
        if ( borderRadiusDirty ) {
          fillElement.style[ Features.borderRadius ] = borderRadius + 'px'; // if one is zero, we are not rounded, so we do the min here
        }
        if ( this.dirtyFill ) {
          fillElement.style.backgroundColor = node.getCSSFill();
        }

        if ( this.dirtyStroke ) {
          // update stroke presence
          if ( node.hasStroke() ) {
            strokeElement.style.borderStyle = 'solid';
          }
          else {
            strokeElement.style.borderStyle = 'none';
          }
        }

        if ( node.hasStroke() ) {
          // since we only execute these if we have a stroke, we need to redo everything if there was no stroke previously.
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only rectangles)
          var hadNoStrokeBefore = !this.hadStroke;

          if ( hadNoStrokeBefore || this.dirtyWidth || this.dirtyLineWidth ) {
            strokeElement.style.width = ( node._rectWidth - node.getLineWidth() ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyHeight || this.dirtyLineWidth ) {
            strokeElement.style.height = ( node._rectHeight - node.getLineWidth() ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = node.getLineWidth() + 'px';
          }

          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = node.getSimpleCSSStroke();
          }

          if ( hadNoStrokeBefore || borderRadiusDirty || this.dirtyLineWidth || this.dirtyLineOptions ) {
            strokeElement.style[ Features.borderRadius ] = ( node.isRounded() || node.getLineJoin() === 'round' ) ? ( borderRadius + node.getLineWidth() / 2 ) + 'px' : '0';
          }
        }
      }

      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyX || this.dirtyY ) {
        scratchMatrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( node._rectX, node._rectY );
        scratchMatrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyPreparedTransform( scratchMatrix, this.fillElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToCleanState();
      this.cleanPaintableState();
      this.transformDirty = false;
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      this.disposeState();

      if ( !keepDOMRectangleElements ) {
        // clear the references
        this.fillElement = null;
        this.strokeElement = null;
        this.domElement = null;
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );

  RectangleStatefulDrawable.mixInto( RectangleDOMDrawable );

  Poolable.mixInto( RectangleDOMDrawable );

  return RectangleDOMDrawable;
} );
