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
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var SVGGroup = require( 'SCENERY/display/SVGGroup' );
  var Util = require( 'SCENERY/util/Util' );
  
  scenery.SVGBlock = function SVGBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  };
  var SVGBlock = scenery.SVGBlock;
  
  inherit( FittedBlock, SVGBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      this.initializeFittedBlock( display, renderer, transformRootInstance );
      
      this.filterRootInstance = filterRootInstance;
      
      this.dirtyGroups = cleanArray( this.dirtyGroups );
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      
      if ( !this.domElement ) {
        // main SVG element
        this.svg = document.createElementNS( scenery.svgns, 'svg' );
        this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
        this.svg.style.position = 'absolute';
        this.svg.style.left = '0';
        this.svg.style.top = '0';
        //OHTWO TODO: why would we clip the individual layers also? Seems like a potentially useless performance loss
        // this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
        this.svg.style['pointer-events'] = 'none';
        
        // the <defs> block that we will be stuffing gradients and patterns into
        this.defs = document.createElementNS( scenery.svgns, 'defs' );
        this.svg.appendChild( this.defs );
        
        this.baseTransformGroup = document.createElementNS( scenery.svgns, 'g' );
        this.svg.appendChild( this.baseTransformGroup );
        
        this.domElement = this.svg;
      }
      
      // reset what layer fitting can do (this.forceAcceleration set in fitted block initialization)
      Util.prepareForTransform( this.svg, this.forceAcceleration );
      Util.unsetTransform( this.svg ); // clear out any transforms that could have been previously applied
      this.baseTransformGroup.setAttribute( 'transform', '' ); // no base transform
      
      var instanceClosestToRoot = transformRootInstance.trail.nodes.length > filterRootInstance.trail.nodes.length ? filterRootInstance : transformRootInstance;
      
      this.rootGroup = SVGGroup.createFromPool( this, instanceClosestToRoot, null );
      this.baseTransformGroup.appendChild( this.rootGroup.svgGroup );
      
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
      
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'initialized #' + this.id );
      
      return this;
    },
    
    markDirtyGroup: function( block ) {
      this.dirtyGroups.push( block );
      this.markDirty();
    },
    
    markDirtyDrawable: function( drawable ) {
      sceneryLog && sceneryLog.dirty && sceneryLog.dirty( 'markDirtyDrawable on SVGBlock#' + this.id + ' with ' + drawable.toString() );
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    setSizeFullDisplay: function() {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'setSizeFullDisplay #' + this.id );
      
      var size = this.display.getSize();
      this.svg.setAttribute( 'width', size.width );
      this.svg.setAttribute( 'height', size.height );
    },
    
    setSizeFitBounds: function() {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'setSizeFitBounds #' + this.id + ' with ' + this.fitBounds.toString() );
      
      var x = this.fitBounds.minX;
      var y = this.fitBounds.minY;
      
      this.baseTransformGroup.setAttribute( 'transform', 'translate(' + (-x) + ',' + (-y) + ')' ); // subtract off so we have a tight fit
      Util.setTransform( 'matrix(1,0,0,1,' + x + ',' + y + ')', this.svg, this.forceAcceleration ); // reapply the translation as a CSS transform
      this.svg.setAttribute( 'width', this.fitBounds.width );
      this.svg.setAttribute( 'height', this.fitBounds.height );
    },
    
    update: function() {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'update #' + this.id );
      
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        //OHTWO TODO: call here!
        while ( this.dirtyGroups.length ) {
          this.dirtyGroups.pop().update();
        }
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
        
        // checks will be done in updateFit() to see whether it is needed
        this.updateFit();
      }
    },
    
    dispose: function() {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'dispose #' + this.id );
      
      // make it take up zero area, so that we don't use up excess memory
      this.svg.setAttribute( 'width', 0 );
      this.svg.setAttribute( 'height', 0 );
      
      // clear references
      this.filterRootInstance = null;
      cleanArray( this.dirtyGroups );
      cleanArray( this.dirtyDrawables );
      
      this.baseTransformGroup.removeChild( this.rootGroup.svgGroup );
      this.rootGroup.dispose();
      this.rootGroup = null;
      
      FittedBlock.prototype.dispose.call( this );
    },
    
    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );
      
      FittedBlock.prototype.addDrawable.call( this, drawable );
      
      SVGGroup.addDrawable( this, drawable );
      drawable.updateDefs( this.defs );
    },
    
    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );
      
      SVGGroup.removeDrawable( this, drawable );
      
      FittedBlock.prototype.removeDrawable.call( this, drawable );
      
      // NOTE: we don't unset the drawable's defs here, since it will either be disposed (will clear it)
      // or will be added to another SVGBlock (which will overwrite it)
    },
    
    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( '#' + this.id + '.onIntervalChange ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );
      
      FittedBlock.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );
    },
    
    toString: function() {
      return 'SVGBlock#' + this.id + '-' + FittedBlock.fitString[this.fit];
    }
  } );
  
  /* jshint -W064 */
  Poolable( SVGBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance, filterRootInstance ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'new from pool' );
          return pool.pop().initialize( display, renderer, transformRootInstance, filterRootInstance );
        } else {
          sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'new from constructor' );
          return new SVGBlock( display, renderer, transformRootInstance, filterRootInstance );
        }
      };
    }
  } );
  
  return SVGBlock;
} );
