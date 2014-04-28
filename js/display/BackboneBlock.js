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
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var CanvasBlock = require( 'SCENERY/display/CanvasBlock' );
  var SVGBlock = require( 'SCENERY/display/SVGBlock' );
  var DOMBlock = require( 'SCENERY/display/DOMBlock' );
  var Util = require( 'SCENERY/util/Util' );
  
  scenery.BackboneBlock = function BackboneBlock( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
    this.initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
  };
  var BackboneBlock = scenery.BackboneBlock;
  
  inherit( Drawable, BackboneBlock, {
    initialize: function( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
      Drawable.call( this, renderer );
      
      this.forceAcceleration = Renderer.isAccelerationForced( this.renderer );
      
      // reference to the instance that controls this backbone
      this.backboneInstance = backboneInstance;
      
      // where is the transform root for our generated blocks?
      this.transformRootInstance = transformRootInstance;
      
      // where have filters been applied to up? our responsibility is to apply filters between this and our backboneInstance
      this.filterRootAncestorInstance = backboneInstance.parent ? backboneInstance.parent.getFilterRootInstance() : backboneInstance;
      
      // where have transforms been applied up to? our responsibility is to apply transforms between this and our backboneInstance
      this.transformRootAncestorInstance = backboneInstance.parent ? backboneInstance.parent.getTransformRootInstance() : backboneInstance;
      
      this.willApplyTransform = this.transformRootAncestorInstance !== this.backboneInstance;
      
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );
      if ( this.willApplyTransform ) {
        this.backboneInstance.addRelativeTransformListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
        this.backboneInstance.addRelativeTransformPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
      }
      
      this.renderer = renderer;
      this.domElement = isDisplayRoot ? display._domElement : BackboneBlock.createDivBackbone();
      this.isDisplayRoot = isDisplayRoot;
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      
      Util.prepareForTransform( this.domElement, this.forceAcceleration );
      
      //OHTWO TODO: listen to backboneInstance, handle visibility if possible (see the filterroot situation?)
      
      this.blocks = this.blocks || []; // we are responsible for their disposal
    },
    
    dispose: function() {
      this.backboneInstance = null;
      this.transformRootInstance = null;
      this.filterRootAncestorInstance = null;
      this.transformRootAncestorInstance = null;
      cleanArray( this.dirtyDrawables );
      
      this.disposeBlocks();
      
      if ( this.willApplyTransform ) {
        this.instance.removeRelativeTransformListener( this.transformListener );
        this.instance.removeRelativeTransformPrecompute();
      }
      
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
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    markTransformDirty: function() {
      assert && assert( this.willApplyTransform, 'Sanity check for willApplyTransform' );
      
      // relative matrix on backbone instance should be up to date, since we added the compute flags
      scenery.Util.applyPreparedTransform( this.backboneInstance.relativeMatrix, this.domElement, this.forceAcceleration );
    },
    
    update: function() {
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
      }
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
            //OHTWO TODO: handle filter root separately from the backbone instance?
            currentBlock = SVGBlock.createFromPool( currentRenderer, this.transformRootInstance, this.backboneInstance );
          } else if ( Renderer.isDOM( currentRenderer ) ) {
            currentBlock = DOMBlock.createFromPool( drawable );
            currentRenderer = 0; // force a new block for the next drawable
          } else {
            throw new Error( 'unsupported renderer for BackboneBlock.rebuild: ' + currentRenderer );
          }
          
          this.blocks.push( currentBlock );
          currentBlock.parentDrawable = this;
          this.domElement.appendChild( currentBlock.domElement ); //OHTWO TODO: minor speedup by appending only once its fragment is constructed? or use DocumentFragment?
          
          // mark it dirty for now, so we can check
          this.markDirtyDrawable( currentBlock );
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
      return function( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
        if ( pool.length ) {
          return pool.pop().initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
        } else {
          return new BackboneBlock( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
        }
      };
    }
  } );
  
  return BackboneBlock;
} );
