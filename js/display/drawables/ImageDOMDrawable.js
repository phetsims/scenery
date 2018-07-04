// Copyright 2016, University of Colorado Boulder

/**
 * DOM drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var ImageStatefulDrawable = require( 'SCENERY/display/drawables/ImageStatefulDrawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Util' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMImageElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of a Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function ImageDOMDrawable( renderer, instance ) {
    // Super-type initialization
    this.initializeDOMSelfDrawable( renderer, instance );

    // Stateful trait initialization
    this.initializeState( renderer, instance );

    // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
    // allocation and performance costs)
    if ( !this.domElement ) {
      // @protected {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
      this.domElement = document.createElement( 'img' );
      this.domElement.style.display = 'block';
      this.domElement.style.position = 'absolute';
      this.domElement.style.pointerEvents = 'none';
      this.domElement.style.left = '0';
      this.domElement.style.top = '0';
    }

    // Whether we have an opacity attribute specified on the DOM element.
    this.hasOpacity = false;

    // Apply CSS needed for future CSS transforms to work properly.
    scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );
  }

scenery.register( 'ImageDOMDrawable', ImageDOMDrawable );

  inherit( DOMSelfDrawable, ImageDOMDrawable, {
    /**
     * Updates our DOM element so that its appearance matches our node's representation.
     * @protected
     *
     * This implements part of the DOMSelfDrawable required API for subtypes.
     */
    updateDOM: function() {
      var node = this.node;
      var img = this.domElement;

      if ( this.paintDirty && this.dirtyImage ) {
        // TODO: allow other ways of showing a DOM image?
        img.src = node._image ? node._image.src : '//:0'; // NOTE: for img with no src (but with a string), see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
      }

      if ( this.dirtyImageOpacity ) {
        if ( node._imageOpacity === 1 ) {
          if ( this.hasOpacity ) {
            this.hasOpacity = false;
            img.style.opacity = '';
          }
        }
        else {
          this.hasOpacity = true;
          img.style.opacity = node._imageOpacity;
        }
      }

      if ( this.transformDirty ) {
        scenery.Util.applyPreparedTransform( this.getTransformMatrix(), this.domElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToCleanState();
      this.transformDirty = false;
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      this.disposeState();

      if ( !keepDOMImageElements ) {
        this.domElement = null; // clear our DOM reference if we want to toss it
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );
  ImageStatefulDrawable.mixInto( ImageDOMDrawable );

  Poolable.mixInto( ImageDOMDrawable );

  return ImageDOMDrawable;
} );
