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
  var Bounds2 = require( 'DOT/Bounds2' );
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
        
        this.baseTransformGroup = document.createElementNS( scenery.svgns, 'g' );
        this.svg.appendChild( this.baseTransformGroup );
        
        this.domElement = this.svg;
      }
      
      // reset what layer fitting can do
      this.svg.style.transform = ''; // no transform
      this.baseTransformGroup.setAttribute( 'transform', '' ); // no base transform
      
      var instanceClosestToRoot = transformRootInstance.trail.nodes.length > filterRootInstance.trail.nodes.length ? filterRootInstance : transformRootInstance;
      
      this.rootGroup = SVGGroup.createFromPool( this, instanceClosestToRoot, null );
      this.baseTransformGroup.appendChild( this.rootGroup.svgGroup );
      
      var canBeFullDisplay = transformRootInstance.state.isDisplayRoot;
      
      //OHTWO TODO: change fit based on renderer flags or extra parameters
      this.fit = canBeFullDisplay ? SVGBlock.fit.FULL_DISPLAY : SVGBlock.fit.COMMON_ANCESTOR;
      
      this.dirtyFit = true;
      this.dirtyFitListener = this.dirtyFitListener || this.markDirtyFit.bind( this );
      this.commonFitInstance = null; // filled in if COMMON_ANCESTOR
      this.fitBounds = Bounds2.NOTHING.copy(); // tracks the "tight" bounds for fitting, not the actually-displayed bounds
      this.oldFitBounds = Bounds2.NOTHING.copy(); // copy for storage
      
      if ( this.fit === SVGBlock.fit.FULL_DISPLAY ) {
        this.display.onStatic( 'displaySize', this.dirtyFitListener );
      }
      
      // TODO: add count of boundsless objects?
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
      
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( 'initialized #' + this.id );
      
      return this;
    },
    
    markDirtyGroup: function( block ) {
      this.dirtyGroups.push( block );
      this.markDirty();
    },
    
    markDirtyDrawable: function( drawable ) {
      sceneryLayerLog && sceneryLayerLog.dirty && sceneryLayerLog.dirty( 'markDirtyDrawable on SVGBlock#' + this.id + ' with ' + drawable.toString() );
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    markDirtyFit: function() {
      sceneryLayerLog && sceneryLayerLog.dirty && sceneryLayerLog.dirty( 'markDirtyFit on SVGBlock#' + this.id );
      this.dirtyFit = true;
      this.markDirty();
    },
    
    update: function() {
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( 'update #' + this.id );
      
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        //OHTWO TODO: call here!
        while ( this.dirtyGroups.length ) {
          this.dirtyGroups.pop().update();
        }
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
        
        // for now, if we aren't full-display, update the fit any time we are "dirty"
        if ( this.dirtyFit || this.fit !== SVGBlock.fit.FULL_DISPLAY ) {
          this.dirtyFit = false;
          this.updateFit();
        }
      }
    },
    
    updateFit: function() {
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( 'updateFit #' + this.id );
      if ( this.fit === SVGBlock.fit.FULL_DISPLAY ) {
        var size = this.display.getSize();
        this.svg.setAttribute( 'width', size.width );
        this.svg.setAttribute( 'height', size.height );
      } else if ( this.fit === SVGBlock.fit.COMMON_ANCESTOR ) {
        assert && assert( this.commonFitInstance.trail.length >= this.transformRootInstance.trail.length );
        
        // will trigger bounds validation (for now) until we have a better way of handling this
        this.fitBounds.set( this.commonFitInstance.node.getLocalBounds() );
        
        //OHTWO TODO: bail out here when possible (should store an old "local" one to compare with?)
        
        // walk it up, transforming so it is relative to our transform root
        var instance = this.commonFitInstance;
        while ( instance !== this.transformRootInstance ) {
          // shouldn't infinite loop, we'll null-pointer beforehand unless something is seriously wrong
          this.fitBounds.transform( instance.node.getMatrix() );
          instance = instance.parent;
        }
        
        //OHTWO TODO: change only when necessary
        if ( !this.fitBounds.equals( this.oldFitBounds ) ) {
          // store our copy for future checks (and do it before we modify this.fitBounds)
          this.oldFitBounds.set( this.fitBounds );
          
          this.fitBounds.roundOut();
          this.fitBounds.dilate( 4 ); // for safety, modify in the future
          
          var x = this.fitBounds.minX;
          var y = this.fitBounds.minY;
          this.baseTransformGroup.setAttribute( 'transform', 'translate(' + (-x) + ',' + (-y) + ')' ); // subtract off so we have a tight fit
          this.svg.style.transform = 'matrix(1,0,0,1,' + x + ',' + y + ')'; // reapply the translation as a CSS transform
          this.svg.setAttribute( 'width', this.fitBounds.width );
          this.svg.setAttribute( 'height', this.fitBounds.height );
        }
      } else {
        throw new Error( 'unknown fit' );
      }
    },
    
    dispose: function() {
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( 'dispose #' + this.id );
      
      if ( this.fit === SVGBlock.fit.FULL_DISPLAY ) {
        this.display.offStatic( 'displaySize', this.dirtyFitListener );
      }
      
      // make it take up zero area, so that we don't use up excess memory
      this.svg.setAttribute( 'width', 0 );
      this.svg.setAttribute( 'height', 0 );
      
      // clear references
      this.transformRootInstance = null;
      this.filterRootInstance = null;
      this.commonFitInstance = null;
      cleanArray( this.dirtyGroups );
      cleanArray( this.dirtyDrawables );
      
      this.baseTransformGroup.removeChild( this.rootGroup.svgGroup );
      this.rootGroup.dispose();
      this.rootGroup = null;
      
      Block.prototype.dispose.call( this );
    },
    
    addDrawable: function( drawable ) {
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );
      
      Block.prototype.addDrawable.call( this, drawable );
      
      SVGGroup.addDrawable( this, drawable );
      drawable.updateDefs( this.defs );
    },
    
    removeDrawable: function( drawable ) {
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );
      
      SVGGroup.removeDrawable( this, drawable );
      drawable.parentDrawable = null;
      
      Block.prototype.removeDrawable.call( this, drawable );
      
      // NOTE: we don't unset the drawable's defs here, since it will either be disposed (will clear it)
      // or will be added to another SVGBlock (which will overwrite it)
    },
    
    notifyInterval: function( firstDrawable, lastDrawable ) {
      sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( '#' + this.id + '.notifyInterval ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );
      
      Block.prototype.notifyInterval.call( this, firstDrawable, lastDrawable );
      
      // if we use a common ancestor fit, find the common ancestor instance
      if ( this.fit === SVGBlock.fit.COMMON_ANCESTOR ) {
        assert && assert( firstDrawable.instance && lastDrawable.instance,
                          'For common-ancestor SVG fits, we need the first and last drawables to have direct instance references' );
        
        var firstInstance = firstDrawable.instance;
        var lastInstance = lastDrawable.instance;
        
        // walk down the longest one until they are a common length
        var minLength = Math.min( firstInstance.trail.length, lastInstance.trail.length );
        while ( firstInstance.trail.length > minLength ) {
          firstInstance = firstInstance.parent;
        }
        while ( lastInstance.trail.length > minLength ) {
          lastInstance = lastInstance.parent;
        }
        
        // step down until they match
        while( firstInstance !== lastInstance ) {
          firstInstance = firstInstance.parent;
          lastInstance = lastInstance.parent;
        }
        
        this.commonFitInstance = firstInstance;
        sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( '   common fit instance: ' + this.commonFitInstance.toString() );
        
        assert && assert( this.commonFitInstance.trail.length >= this.transformRootInstance.trail.length );
        
        this.markDirtyFit();
      }
    },
    
    toString: function() {
      return 'SVGBlock#' + this.id + ' ' + SVGBlock.fitString[this.fit];
    }
  } );
  
  SVGBlock.fit = {
    FULL_DISPLAY: 1,
    COMMON_ANCESTOR: 2
  };
  SVGBlock.fitString = {
    1: 'fullDisplay',
    2: 'commonAncestor'
  };
  
  /* jshint -W064 */
  Poolable( SVGBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance, filterRootInstance ) {
        if ( pool.length ) {
          sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( 'new from pool' );
          return pool.pop().initialize( display, renderer, transformRootInstance, filterRootInstance );
        } else {
          sceneryLayerLog && sceneryLayerLog.SVGBlock && sceneryLayerLog.SVGBlock( 'new from constructor' );
          return new SVGBlock( display, renderer, transformRootInstance, filterRootInstance );
        }
      };
    }
  } );
  
  return SVGBlock;
} );
