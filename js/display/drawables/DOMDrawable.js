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
    // Super-type initialization
    this.initializeDOMSelfDrawable( renderer, instance );

    // @public {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
    this.domElement = this.node._container;

    // Apply CSS needed for future CSS transforms to work properly.
    scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );
  }

  scenery.register( 'DOMDrawable', DOMDrawable );

  inherit( DOMSelfDrawable, DOMDrawable, {
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

  Poolable.mixInto( DOMDrawable );

  return DOMDrawable;
} );
