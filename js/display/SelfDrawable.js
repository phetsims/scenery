// Copyright 2014-2016, University of Colorado Boulder


/**
 * A drawable that will paint a single instance.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Drawable = require( 'SCENERY/display/Drawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   * @extends Drawable
   * @mixes Poolable
   *
   * @param renderer
   * @param instance
   */
  function SelfDrawable( renderer, instance ) {
    this.initializeSelfDrawable( renderer, instance );
  }

  scenery.register( 'SelfDrawable', SelfDrawable );

  inherit( scenery.Drawable, SelfDrawable, {
    initializeSelfDrawable: function( renderer, instance ) {
      this.drawableVisibilityListener = this.drawableVisibilityListener || this.updateSelfVisibility.bind( this );

      // super initialization
      this.initializeDrawable( renderer );

      this.instance = instance;
      this.node = instance.trail.lastNode();
      this.node.attachDrawable( this );

      this.instance.onStatic( 'selfVisibility', this.drawableVisibilityListener );

      this.updateSelfVisibility();

      return this;
    },

    // @public
    dispose: function() {
      this.instance.offStatic( 'selfVisibility', this.drawableVisibilityListener );

      this.node.detachDrawable( this );

      // free references
      this.instance = null;
      this.node = null;

      Drawable.prototype.dispose.call( this );
    },

    updateSelfVisibility: function() {
      // hide our drawable if it is not relatively visible
      this.visible = this.instance.selfVisible;
    },

    toDetailedString: function() {
      return this.toString() + ' (' + this.instance.trail.toPathString() + ')';
    }
  } );

  return SelfDrawable;
} );
