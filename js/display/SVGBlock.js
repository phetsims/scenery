// Copyright 2002-2014, University of Colorado

/**
 * Handles a visual SVG layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var Block = require( 'SCENERY/display/Block' );
  var SVGGroup = require( 'SCENERY/display/SVGGroup' );
  
  scenery.SVGBlock = function SVGBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  };
  var SVGBlock = scenery.SVGBlock;
  
  inherit( Block, SVGBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      this.initializeBlock( display, renderer );
      
      this.transformRootInstance = transformRootInstance;
      this.filterRootInstance = filterRootInstance;
      
      this.dirtyGroups = cleanArray( this.dirtyGroups );
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      
      if ( !this.domElement ) {
        // main SVG element
        this.svg = document.createElementNS( scenery.svgns, 'svg' );
        // this.svg.setAttribute( 'width', width );
        // this.svg.setAttribute( 'height', height );
        this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
        this.svg.style.position = 'absolute';
        this.svg.style.left = '0';
        this.svg.style.top = '0';
        // this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
        this.svg.style['pointer-events'] = 'none';
        
        // the <defs> block that we will be stuffing gradients and patterns into
        this.defs = document.createElementNS( scenery.svgns, 'defs' );
        this.svg.appendChild( this.defs );
        
        this.domElement = this.svg;
      }
      
      var instanceClosestToRoot = transformRootInstance.trail.nodes.length > filterRootInstance.trail.nodes.length ? filterRootInstance : transformRootInstance;
      
      this.rootGroup = SVGGroup.createFromPool( this, instanceClosestToRoot, null );
      this.svg.appendChild( this.rootGroup.svgGroup );
      
      var canBeFullDisplay = transformRootInstance.state.isDisplayRoot;
      
      //OHTWO TODO: change fit based on renderer flags or extra parameters
      this.fit = canBeFullDisplay ? SVGBlock.fit.FULL_DISPLAY : SVGBlock.fit.COMMON_ANCESTOR;
      
      this.dirtyFit = true;
      this.dirtyFitListener = this.dirtyFitListener || this.markDirtyFit.bind( this );
      
      if ( this.fit === SVGBlock.fit.FULL_DISPLAY ) {
        this.display.onStatic( 'displaySize', this.dirtyFitListener );
      }
      
      // TODO: add count of boundsless objects?
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
      
      return this;
    },
    
    markDirtyGroup: function( block ) {
      this.dirtyGroups.push( block );
      this.markDirty();
    },
    
    markDirtyDrawable: function( drawable ) {
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    markDirtyFit: function() {
      this.dirtyFit = true;
      this.markDirty();
    },
    
    update: function() {
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        //OHTWO TODO: call here!
        while ( this.dirtyGroups.length ) {
          this.dirtyGroups.pop().update();
        }
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
        
        if ( this.dirtyFit ) {
          this.dirtyFit = false;
          this.updateFit();
        }
      }
    },
    
    updateFit: function() {
      if ( this.fit === SVGBlock.fit.FULL_DISPLAY ) {
        var size = this.display.getSize();
        this.svg.setAttribute( 'width', size.width );
        this.svg.setAttribute( 'height', size.height );
      } else if ( this.fit === SVGBlock.fit.COMMON_ANCESTOR ) {
        
      } else {
        throw new Error( 'unknown fit' );
      }
    },
    
    dispose: function() {
      if ( this.fit === SVGBlock.fit.FULL_DISPLAY ) {
        this.display.offStatic( 'displaySize', this.dirtyFitListener );
      }
      
      // clear references
      this.transformRootInstance = null;
      this.filterRootInstance = null;
      cleanArray( this.dirtyGroups );
      cleanArray( this.dirtyDrawables );
      
      this.svg.removeChild( this.rootGroup.svgGroup );
      this.rootGroup.dispose();
      this.rootGroup = null;
      
      Block.prototype.dispose.call( this );
    },
    
    addDrawable: function( drawable ) {
      Block.prototype.addDrawable.call( this, drawable );
      
      SVGGroup.addDrawable( this, drawable );
      drawable.updateDefs( this.defs );
    },
    
    removeDrawable: function( drawable ) {
      SVGGroup.removeDrawable( this, drawable );
      drawable.parentDrawable = null;
      
      Block.prototype.removeDrawable.call( this, drawable );
      
      // NOTE: we don't unset the drawable's defs here, since it will either be disposed (will clear it)
      // or will be added to another SVGBlock (which will overwrite it)
    }
  } );
  
  SVGBlock.fit = {
    FULL_DISPLAY: 1,
    COMMON_ANCESTOR: 2
  };
  
  /* jshint -W064 */
  Poolable( SVGBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance, filterRootInstance ) {
        if ( pool.length ) {
          return pool.pop().initialize( display, renderer, transformRootInstance, filterRootInstance );
        } else {
          return new SVGBlock( display, renderer, transformRootInstance, filterRootInstance );
        }
      };
    }
  } );
  
  return SVGBlock;
} );
