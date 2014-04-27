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
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var CanvasBlock = require( 'SCENERY/display/CanvasBlock' );
  var SVGBlock = require( 'SCENERY/display/SVGBlock' );
  var DOMBlock = require( 'SCENERY/display/DOMBlock' );
  
  scenery.BackboneBlock = function BackboneBlock( backboneInstance, transformRootInstance, renderer, isDisplayRoot, existingDiv ) {
    this.initialize( backboneInstance, transformRootInstance, renderer, isDisplayRoot, existingDiv );
  };
  var BackboneBlock = scenery.BackboneBlock;
  
  inherit( Drawable, BackboneBlock, {
    initialize: function( backboneInstance, transformRootInstance, renderer, isDisplayRoot, existingDiv ) {
      Drawable.call( this, renderer );
      
      this.backboneInstance = backboneInstance;
      this.transformRootInstance = transformRootInstance;
      this.renderer = renderer;
      this.domElement = existingDiv || BackboneBlock.createDivBackbone();
      this.isDisplayRoot = isDisplayRoot;
      
      this.blocks = this.blocks || []; // we are responsible for their disposal
    },
    
    dispose: function() {
      this.backboneInstance = null;
      this.transformRootInstance = null;
      
      this.disposeBlocks();
      
      Drawable.prototype.dispose.call( this );
    },
    
    // dispose all of the blocks while clearing our references to them
    disposeBlocks: function() {
      while ( this.blocks.length ) {
        var block = this.blocks.pop();
        this.domElement.removeChild( block.domElement );
        block.dispose();
      }
    },
    
    markDirtyDrawable: function( drawable ) {
      
    },
    
    rebuild: function( firstDrawable, lastDrawable ) {
      this.disposeBlocks();
      
      var currentBlock = null;
      var currentRenderer = 0;
      
      // linked-list iteration inclusively from firstDrawable to lastDrawable
      for ( var drawable = firstDrawable; drawable !== null && drawable.previousDrawable !== lastDrawable; drawable = drawable.nextDrawable ) {
        
        // if we need to switch to a new block, create it
        if ( !currentBlock || drawable.renderer !== currentRenderer ) {
          currentRenderer = drawable.renderer;
          
          if ( Renderer.isCanvas( currentRenderer ) ) {
            currentBlock = CanvasBlock.createFromPool( currentRenderer, this.transformRootInstance );
          } else if ( Renderer.isSVG( currentRenderer ) ) {
            currentBlock = SVGBlock.createFromPool( currentRenderer, this.transformRootInstance );
          } else if ( Renderer.isDOM( currentRenderer ) ) {
            currentBlock = DOMBlock.createFromPool( drawable );
            currentRenderer = 0; // force a new block for the next drawable
          } else {
            throw new Error( 'unsupported renderer for BackboneBlock.rebuild: ' + currentRenderer );
          }
          
          this.blocks.push( currentBlock );
          this.domElement.appendChild( currentBlock.domElement ); //OHTWO TODO: minor speedup by appending only once its fragment is constructed? or use DocumentFragment?
        }
        
        currentBlock.addDrawable( drawable );
      }
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
      return function( backboneInstance, transformRootInstance, renderer, isDisplayRoot, existingDiv ) {
        if ( pool.length ) {
          return pool.pop().initialize( backboneInstance, transformRootInstance, renderer, isDisplayRoot, existingDiv );
        } else {
          return new BackboneBlock( backboneInstance, transformRootInstance, renderer, isDisplayRoot, existingDiv );
        }
      };
    }
  } );
  
  return BackboneBlock;
} );
