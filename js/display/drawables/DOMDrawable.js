// Copyright 2016, University of Colorado Boulder

/**
 * DOM renderer for DOM nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Util' );

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our DOM node. One of these drawables will be created
   * for each displayed instance of a DOM node.
   * @public (scenery-internal)
   * @constructor
   * @extends DOMSelfDrawable
   * @mixes SelfDrawable.Poolable
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function DOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }

  scenery.register( 'DOMDrawable', DOMDrawable );

  inherit( DOMSelfDrawable, DOMDrawable, {
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
     * @returns {DOMDrawable} - Self reference for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeDOMSelfDrawable( renderer, instance );

      // @public {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
      this.domElement = this.node._container;

      // Apply CSS needed for future CSS transforms to work properly.
      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

      return this; // allow for chaining
    },

    /**
     * Updates our DOM element so that its appearance matches our node's representation.
     * @protected
     *
     * This implements part of the DOMSelfDrawable required API for subtypes.
     */
    updateDOM: function() {
      if ( this.transformDirty && !this.node._preventTransform ) {
        scenery.Util.applyPreparedTransform( this.getTransformMatrix(), this.domElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.transformDirty = false;
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      DOMSelfDrawable.prototype.dispose.call( this );

      this.domElement = null;
    }
  } );

  Poolable.mixInto( DOMDrawable, {
    initialize: DOMDrawable.prototype.initialize
  } );

  return DOMDrawable;
} );
