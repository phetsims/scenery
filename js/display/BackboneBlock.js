// Copyright 2002-2014, University of Colorado

/**
 * A "backbone" block that controls a DOM element (usually a div) that contains other blocks with DOM/SVG/Canvas/WebGL content
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  // includeRoot is used for the root of a display, where the instance should be thought of as fully "under" the backbone
  scenery.BackboneBlock = function BackboneBlock( instance, renderer, includeRoot, existingDiv ) {
    this.initialize( instance, renderer, includeRoot, existingDiv );
  };
  var BackboneBlock = scenery.BackboneBlock;
  
  inherit( Drawable, BackboneBlock, {
    initialize: function( instance, renderer, includeRoot, existingDiv ) {
      Drawable.call( this, renderer );
      
      this.instance = instance;
      this.renderer = renderer;
      this.domElement = existingDiv || BackboneBlock.createDivBackbone();
      this.includeRoot = includeRoot;
    },
    
    dispose: function() {
      this.instance = null;
      
      Drawable.prototype.dispose.call( this );
    },
    
    markDirtyDrawable: function( drawable ) {
      
    }
  } );
  
  BackboneBlock.createDivBackbone = function() {
    var div = document.createElement( 'div' );
    div.style.position = 'absolute';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '0';
    div.style.height = '0';
    return div;
  };
  
  /* jshint -W064 */
  Poolable( BackboneBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( instance, renderer, includeRoot, existingDiv ) {
        if ( pool.length ) {
          return pool.pop().initialize( instance, renderer, includeRoot, existingDiv );
        } else {
          return new BackboneBlock( instance, renderer, includeRoot, existingDiv );
        }
      };
    }
  } );
  
  return BackboneBlock;
} );
