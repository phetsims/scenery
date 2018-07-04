// Copyright 2016, University of Colorado Boulder

/**
 * DOM drawable for Circle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CircleStatefulDrawable = require( 'SCENERY/display/drawables/CircleStatefulDrawable' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var Features = require( 'SCENERY/util/Features' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Util' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMCircleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our Circle. One of these drawables will be created
   * for each displayed instance of a Circle.
   * @public (scenery-internal)
   * @constructor
   * @extends DOMSelfDrawable
   * @mixes Circle.CircleStatefulDrawable
   * @mixes Paintable.PaintableStatefulDrawable
   * @mixes SelfDrawable.Poolable
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function CircleDOMDrawable( renderer, instance ) {
    // Super-type initialization
    this.initializeDOMSelfDrawable( renderer, instance );

    // Stateful trait initialization
    this.initializeState( renderer, instance );

    // @protected {Matrix3} - We need to store an independent matrix, as our CSS transform actually depends on the radius.
    this.matrix = this.matrix || Matrix3.dirtyFromPool();

    // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
    // allocation and performance costs)
    if ( !this.fillElement || !this.strokeElement ) {
      // @protected {HTMLDivElement} - Will contain the fill by manipulating borderRadius
      var fillElement = this.fillElement = document.createElement( 'div' );

      // @protected {HTMLDivElement} - Will contain the stroke by manipulating borderRadius
      var strokeElement = this.strokeElement = document.createElement( 'div' );

      fillElement.style.display = 'block';
      fillElement.style.position = 'absolute';
      fillElement.style.left = '0';
      fillElement.style.top = '0';
      fillElement.style.pointerEvents = 'none';
      strokeElement.style.display = 'block';
      strokeElement.style.position = 'absolute';
      strokeElement.style.left = '0';
      strokeElement.style.top = '0';
      strokeElement.style.pointerEvents = 'none';

      // Nesting allows us to transform only one AND to guarantee that the stroke is on top.
      fillElement.appendChild( strokeElement );
    }

    // @protected {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
    this.domElement = this.fillElement;

    // Apply CSS needed for future CSS transforms to work properly.
    scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );
  }

  scenery.register( 'CircleDOMDrawable', CircleDOMDrawable );

  inherit( DOMSelfDrawable, CircleDOMDrawable, {
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

      // If paintDirty is false, there are no updates that are needed.
      if ( this.paintDirty ) {
        if ( this.dirtyRadius ) {
          fillElement.style.width = ( 2 * node._radius ) + 'px';
          fillElement.style.height = ( 2 * node._radius ) + 'px';
          fillElement.style[ Features.borderRadius ] = node._radius + 'px';
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
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only Circles)
          var hadNoStrokeBefore = !this.hadStroke;

          if ( hadNoStrokeBefore || this.dirtyLineWidth || this.dirtyRadius ) {
            strokeElement.style.width = ( 2 * node._radius - node.getLineWidth() ) + 'px';
            strokeElement.style.height = ( 2 * node._radius - node.getLineWidth() ) + 'px';
            strokeElement.style[ Features.borderRadius ] = ( node._radius + node.getLineWidth() / 2 ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = node.getLineWidth() + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = node.getSimpleCSSStroke();
          }
        }
      }

      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyRadius ) {
        this.matrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( -node._radius, -node._radius );
        this.matrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyPreparedTransform( this.matrix, this.fillElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToCleanState();
      this.cleanPaintableState();
      this.transformDirty = false;
    },

    /**
     * Disposes the drawable.
     * @public (scenery-internal)
     * @override
     */
    dispose: function() {
      this.disposeState();

      // Release the DOM elements from the poolable visual state so they aren't kept in memory.
      // May not be done on platforms where we have enough memory to pool these
      if ( !keepDOMCircleElements ) {
        // clear the references
        this.fillElement = null;
        this.strokeElement = null;
        this.domElement = null;
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );

  // Include Circle's stateful trait (used for dirty flags)
  CircleStatefulDrawable.mixInto( CircleDOMDrawable );

  Poolable.mixInto( CircleDOMDrawable );

  return CircleDOMDrawable;
} );
