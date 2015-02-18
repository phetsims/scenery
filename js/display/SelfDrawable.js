// Copyright 2002-2014, University of Colorado Boulder


/**
 * A drawable that will paint a single instance.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );

  scenery.SelfDrawable = function SelfDrawable( renderer, instance ) {
    this.initializeSelfDrawable( renderer, instance );
  };
  var SelfDrawable = scenery.SelfDrawable;

  inherit( scenery.Drawable, SelfDrawable, {
    initializeSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeDrawable( renderer );

      this.instance = instance;
      this.node = instance.trail.lastNode();
      this.node.attachDrawable( this );

      return this;
    },

    // @public
    dispose: function() {
      this.node.detachDrawable( this );

      // free references
      this.instance = null;
      this.node = null;

      Drawable.prototype.dispose.call( this );
    },

    toDetailedString: function() {
      return this.toString() + ' (' + this.instance.trail.toPathString() + ')';
    }
  } );

  SelfDrawable.Poolable = {
    mixin: function( selfDrawableType ) {
      // for pooling, allow <SelfDrawableType>.createFromPool( renderer, instance ) and drawable.freeToPool(). Creation will initialize the drawable to an initial state
      Poolable.mixin( selfDrawableType, {
        defaultFactory: function() {
          /* jshint -W055 */
          return new selfDrawableType();
        },
        constructorDuplicateFactory: function( pool ) {
          return function( renderer, instance ) {
            if ( pool.length ) {
              return pool.pop().initialize( renderer, instance );
            }
            else {
              /* jshint -W055 */
              return new selfDrawableType( renderer, instance );
            }
          };
        }
      } );
    }
  };

  return SelfDrawable;
} );
