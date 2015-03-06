// Copyright 2002-2014, University of Colorado Boulder


/**
 * Handles a visual Pixi layer of drawables.
 *
 * STATUS REPORT
 * March 4 2015
 * PixiBlock is in bad condition--it is a collection of prototypes and not ready for production code.  The best way to
 * see the status of PixiBlock is to launch:
 * http://localhost/forces-and-motion-basics/forces-and-motion-basics_en.html?rootRenderer=pixi&screens=1
 *
 * Completed in this version of forces and motion: basics, with pixi
 * 1. Images are rendering, and are in the correct position
 *
 * Issues in this version of forces and motion: basics, with pixi
 * 1. Z-ordering is incorrect.  When dragging a (hidden) puller, the z-order of many things changes.
 * 2. When dropping a puller, there is an exception that crashes the simulation
 * 3. Resizing is not handled
 * 4. Context loss is not handled
 * 5. Path is not generalized--just doing moveTo/lineTo
 * 6. Getting 30 fps when dragging a puller on iPad3. (Also note that transparency, curves, etc will slow this down further)
 *      Though we could take steps to optimize for pixi, such as batching images together
 *
 * @author Sam Reid
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var count = 0;
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
        this.stage = new PIXI.Stage();

        // Create the renderer and view
        // Size will be set in update
        this.pixiRenderer = PIXI.autoDetectRenderer( 0, 0, {
          transparent: true,
          preserveDrawingBuffer: this.display.options.preserveDrawingBuffer // major performance hit if true
        } );

        // main DOM element
        this.pixiCanvas = this.pixiRenderer.view;
        this.pixiCanvas.style.position = 'absolute';
        this.pixiCanvas.style.left = '0';
        this.pixiCanvas.style.top = '0';
        //OHTWO TODO: why would we clip the individual layers also? Seems like a potentially useless performance loss
        // this.pixiDisplayObject.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
        this.pixiCanvas.style[ 'pointer-events' ] = 'none';
        this.canvasId = this.pixiCanvas.id = 'scenery-pixi' + this.id;

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
      console.log( 'who is calling this code?' );
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
      this.pixiRenderer.resize( this.fitBounds.width, this.fitBounds.height );
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
        // hack to prevent updateFit() from calling all the time and destroying performance
        if ( count % 1000 === 0 ) {
          this.updateFit();
        }
        count++;
        this.pixiRenderer.render( this.stage );
      }
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
