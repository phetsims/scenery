// Copyright 2002-2014, University of Colorado Boulder


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
  var PixiDisplayObject = require( 'SCENERY/display/PixiDisplayObject' );
  var Util = require( 'SCENERY/util/Util' );

  scenery.PixiBlock = function PixiBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  };
  var PixiBlock = scenery.PixiBlock;

  inherit( FittedBlock, PixiBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      this.initializeFittedBlock( display, renderer, transformRootInstance );

      this.filterRootInstance = filterRootInstance;

      this.dirtyGroups = cleanArray( this.dirtyGroups );
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      this.paintMap = {}; // maps {string} paint.id => { count: {number}, paint: {Paint}, def: {SVGElement} }

      if ( !this.domElement ) {

        // Create the Pixi Stage
        this.stage = new PIXI.Stage( 0xFF0000 );

        // Create the renderer and view
        this.pixiRenderer = PIXI.autoDetectRenderer( 1024, 768, { transparent: false } );

        // main DOM element
        this.pixiCanvas = this.pixiRenderer.view;
        this.pixiCanvas.style.position = 'absolute';
        this.pixiCanvas.style.left = '0';
        this.pixiCanvas.style.top = '0';
        //OHTWO TODO: why would we clip the individual layers also? Seems like a potentially useless performance loss
        // this.pixiDisplayObject.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
        this.pixiCanvas.style[ 'pointer-events' ] = 'none';

        this.baseTransformGroup = new PIXI.DisplayObjectContainer();
        this.stage.addChild( this.baseTransformGroup );
        this.domElement = this.pixiCanvas;
      }

      // reset what layer fitting can do (this.forceAcceleration set in fitted block initialization)
      Util.prepareForTransform( this.pixiCanvas, this.forceAcceleration );
      Util.unsetTransform( this.pixiCanvas ); // clear out any transforms that could have been previously applied

      var instanceClosestToRoot = transformRootInstance.trail.nodes.length > filterRootInstance.trail.nodes.length ? filterRootInstance : transformRootInstance;

      this.rootGroup = PixiDisplayObject.createFromPool( this, instanceClosestToRoot, null );
      this.baseTransformGroup.addChild( this.rootGroup.displayObject );

      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)

      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'initialized #' + this.id );

      return this;
    },

    markDirtyGroup: function( block ) {
      this.dirtyGroups.push( block );
      this.markDirty();
    },

    markDirtyDrawable: function( drawable ) {
      sceneryLog && sceneryLog.dirty && sceneryLog.dirty( 'markDirtyDrawable on PixiBlock#' + this.id + ' with ' + drawable.toString() );
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },

    setSizeFullDisplay: function() {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'setSizeFullDisplay #' + this.id );

      var size = this.display.getSize();
      this.pixiCanvas.setAttribute( 'width', size.width );
      this.pixiCanvas.setAttribute( 'height', size.height );
    },

    setSizeFitBounds: function() {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'setSizeFitBounds #' + this.id + ' with ' + this.fitBounds.toString() );

      var x = this.fitBounds.minX;
      var y = this.fitBounds.minY;

      // subtract off so we have a tight fit
      this.baseTransformGroup.x = (-x);
      this.baseTransformGroup.y = (-y);
      Util.setTransform( 'matrix(1,0,0,1,' + x + ',' + y + ')', this.pixiCanvas, this.forceAcceleration ); // reapply the translation as a CSS transform
      this.pixiCanvas.setAttribute( 'width', this.fitBounds.width );
      this.pixiCanvas.setAttribute( 'height', this.fitBounds.height );
    },

    update: function() {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'update #' + this.id );

      if ( this.dirty && !this.disposed ) {
        this.dirty = false;

        //OHTWO TODO: call here!
        while ( this.dirtyGroups.length ) {
          var group = this.dirtyGroups.pop();

          // if this group has been disposed or moved to another block, don't mess with it
          if ( group.block === this ) {
            group.update();
          }
        }
        while ( this.dirtyDrawables.length ) {
          var drawable = this.dirtyDrawables.pop();

          // if this drawable has been disposed or moved to another block, don't mess with it
          if ( drawable.parentDrawable === this ) {
            drawable.update();
          }
        }

        // checks will be done in updateFit() to see whether it is needed
        this.updateFit();

      }
      this.pixiRenderer.render( this.stage );
    },

    dispose: function() {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'dispose #' + this.id );

      // make it take up zero area, so that we don't use up excess memory
      this.pixiCanvas.setAttribute( 'width', 0 );
      this.pixiCanvas.setAttribute( 'height', 0 );

      // clear references
      this.filterRootInstance = null;
      cleanArray( this.dirtyGroups );
      cleanArray( this.dirtyDrawables );
      this.paintMap = {};

      this.baseTransformGroup.removeChild( this.rootGroup.PixiDisplayObject );
      this.rootGroup.dispose();
      this.rootGroup = null;

      FittedBlock.prototype.dispose.call( this );
    },

    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );

      FittedBlock.prototype.addDrawable.call( this, drawable );

      PixiDisplayObject.addDrawable( this, drawable );
      drawable.updatePixiBlock( this );
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );

      PixiDisplayObject.removeDrawable( this, drawable );

      FittedBlock.prototype.removeDrawable.call( this, drawable );

      // NOTE: we don't unset the drawable's defs here, since it will either be disposed (will clear it)
      // or will be added to another PixiBlock (which will overwrite it)
    },

    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( '#' + this.id + '.onIntervalChange ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );

      FittedBlock.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );
    },

    toString: function() {
      return 'PixiBlock#' + this.id + '-' + FittedBlock.fitString[ this.fit ];
    }
  } );

  Poolable.mixin( PixiBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance, filterRootInstance ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'new from pool' );
          return pool.pop().initialize( display, renderer, transformRootInstance, filterRootInstance );
        }
        else {
          sceneryLog && sceneryLog.PixiBlock && sceneryLog.PixiBlock( 'new from constructor' );
          return new PixiBlock( display, renderer, transformRootInstance, filterRootInstance );
        }
      };
    }
  } );

  return PixiBlock;
} );
