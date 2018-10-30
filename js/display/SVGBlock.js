// Copyright 2013-2017, University of Colorado Boulder

/**
 * Handles a visual SVG layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGGroup = require( 'SCENERY/display/SVGGroup' );
  var Util = require( 'SCENERY/util/Util' );

  /**
   * @constructor
   * @extends FittedBlock
   * @mixes Poolable
   *
   * @param {Display} display - the scenery Display this SVGBlock will appear in
   * @param {number} renderer - the bitmask for the renderer, see Renderer.js
   * @param {Instance} transformRootInstance - TODO: Documentation
   * @param {Instance} filterRootInstance - TODO: Documentation
   * @constructor
   */
  function SVGBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  }

  scenery.register( 'SVGBlock', SVGBlock );

  inherit( FittedBlock, SVGBlock, {

    /**
     * Initialize function, which is required since SVGBlock instances are pooled by scenery.
     *
     * @param {Display} display - the scenery Display this SVGBlock will appear in
     * @param {number} renderer - the bitmask for the renderer, see Renderer.js
     * @param {Instance} transformRootInstance - TODO: Documentation
     * @param {Instance} filterRootInstance - TODO: Documentation
     * @returns {FittedBlock}
     */
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      this.initializeFittedBlock( display, renderer, transformRootInstance, FittedBlock.COMMON_ANCESTOR );

      this.filterRootInstance = filterRootInstance;

      this.dirtyGradients = cleanArray( this.dirtyGradients );
      this.dirtyGroups = cleanArray( this.dirtyGroups );
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      // Keep track of how many times each Paint is used in this SVGBlock so that when all usages have been eliminated
      // we can remove the SVG def from our SVG tree to prevent memory leaks, etc.
      // maps {string} paint.id => { count: {number}, paint: {Paint}, def: {SVGElement} }
      // @private
      this.paintMap = {};

      if ( !this.domElement ) {

        // main SVG element
        this.svg = document.createElementNS( scenery.svgns, 'svg' );
        this.svg.style.position = 'absolute';
        this.svg.style.left = '0';
        this.svg.style.top = '0';

        // a11y - make sure the element is not focusable (it is focusable by default in IE11 full screen mode)
        this.svg.setAttribute( 'focusable', false );

        //OHTWO TODO: why would we clip the individual layers also? Seems like a potentially useless performance loss
        // this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
        this.svg.style[ 'pointer-events' ] = 'none';

        // the <defs> block that we will be stuffing gradients and patterns into
        this.defs = document.createElementNS( scenery.svgns, 'defs' );
        this.svg.appendChild( this.defs );

        this.baseTransformGroup = document.createElementNS( scenery.svgns, 'g' );
        this.svg.appendChild( this.baseTransformGroup );

        this.domElement = this.svg;
      }

      // reset what layer fitting can do (this.forceAcceleration set in fitted block initialization)
      Util.prepareForTransform( this.svg, this.forceAcceleration ); // Apply CSS needed for future CSS transforms to work properly.

      Util.unsetTransform( this.svg ); // clear out any transforms that could have been previously applied
      this.baseTransformGroup.setAttribute( 'transform', '' ); // no base transform

      var instanceClosestToRoot = transformRootInstance.trail.nodes.length > filterRootInstance.trail.nodes.length ?
                                  filterRootInstance : transformRootInstance;

      this.rootGroup = SVGGroup.createFromPool( this, instanceClosestToRoot, null );
      this.baseTransformGroup.appendChild( this.rootGroup.svgGroup );

      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)

      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'initialized #' + this.id );

      return this;
    },

    /*
     * Increases our reference count for the specified {Paint}. If it didn't exist before, we'll add the SVG def to the
     * paint can be referenced by SVG id.
     *
     * @param {Paint} paint
     */
    incrementPaint: function( paint ) {
      assert && assert( paint.isPaint );

      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( 'incrementPaint ' + this.toString() + ' ' + paint.id );

      if ( this.paintMap.hasOwnProperty( paint.id ) ) {
        this.paintMap[ paint.id ].count++;
      }
      else {
        var svgPaint = paint.createSVGPaint( this );
        svgPaint.definition.setAttribute( 'id', paint.id + '-' + this.id );

        // TODO: reduce allocations? (pool these)
        this.paintMap[ paint.id ] = {
          count: 1,
          paint: paint,
          svgPaint: svgPaint
        };

        this.defs.appendChild( svgPaint.definition );
      }
    },

    /*
     * Decreases our reference count for the specified {Paint}. If this was the last reference, we'll remove the SVG def
     * from our SVG tree to prevent memory leaks, etc.
     *
     * @param {Paint} paint
     */
    decrementPaint: function( paint ) {
      assert && assert( paint.isPaint );

      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( 'decrementPaint ' + this.toString() + ' ' + paint.id );

      // since the block may have been disposed (yikes!), we have a defensive set-up here
      if ( this.paintMap.hasOwnProperty( paint.id ) ) {
        var entry = this.paintMap[ paint.id ];
        assert && assert( entry.count >= 1 );

        if ( entry.count === 1 ) {
          this.defs.removeChild( entry.svgPaint.definition );
          entry.svgPaint.dispose();
          delete this.paintMap[ paint.id ]; // delete, so we don't memory leak if we run through MANY paints
        }
        else {
          entry.count--;
        }
      }
    },

    markDirtyGradient: function( gradient ) {
      this.dirtyGradients.push( gradient );
      this.markDirty();
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

      this.baseTransformGroup.removeAttribute( 'transform' );
      Util.unsetTransform( this.svg );

      var size = this.display.getSize();
      this.svg.setAttribute( 'width', size.width );
      this.svg.setAttribute( 'height', size.height );
    },

    setSizeFitBounds: function() {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'setSizeFitBounds #' + this.id + ' with ' + this.fitBounds.toString() );

      var x = this.fitBounds.minX;
      var y = this.fitBounds.minY;

      assert && assert( isFinite( x ) && isFinite( y ), 'Invalid SVG transform for SVGBlock' );
      assert && assert( this.fitBounds.isValid(), 'Invalid fitBounds' );

      this.baseTransformGroup.setAttribute( 'transform', 'translate(' + (-x) + ',' + (-y) + ')' ); // subtract off so we have a tight fit
      Util.setTransform( 'matrix(1,0,0,1,' + x + ',' + y + ')', this.svg, this.forceAcceleration ); // reapply the translation as a CSS transform
      this.svg.setAttribute( 'width', this.fitBounds.width );
      this.svg.setAttribute( 'height', this.fitBounds.height );
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !FittedBlock.prototype.update.call( this ) ) {
        return false;
      }

      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'update #' + this.id );

      //OHTWO TODO: call here!
      // TODO: What does the above TODO mean?
      while ( this.dirtyGroups.length ) {
        var group = this.dirtyGroups.pop();

        // if this group has been disposed or moved to another block, don't mess with it
        if ( group.block === this ) {
          group.update();
        }
      }
      while ( this.dirtyGradients.length ) {
        this.dirtyGradients.pop().update();
      }
      while ( this.dirtyDrawables.length ) {
        var drawable = this.dirtyDrawables.pop();

        // if this drawable has been disposed or moved to another block, don't mess with it
        // TODO: If it was moved to another block, why might it still appear in our list?  Shouldn't that be an assertion check?
        if ( drawable.parentDrawable === this ) {
          drawable.update();
        }
      }

      // checks will be done in updateFit() to see whether it is needed
      this.updateFit();

      return true;
    },

    dispose: function() {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( 'dispose #' + this.id );

      // make it take up zero area, so that we don't use up excess memory
      this.svg.setAttribute( 'width', 0 );
      this.svg.setAttribute( 'height', 0 );

      // clear references
      this.filterRootInstance = null;
      cleanArray( this.dirtyGradients );
      cleanArray( this.dirtyGroups );
      cleanArray( this.dirtyDrawables );
      this.paintMap = {};

      this.baseTransformGroup.removeChild( this.rootGroup.svgGroup );
      this.rootGroup.dispose();
      this.rootGroup = null;

      // since we may not properly remove all defs yet
      while ( this.defs.childNodes.length ) {
        this.defs.removeChild( this.defs.childNodes[ 0 ] );
      }

      FittedBlock.prototype.dispose.call( this );
    },

    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );

      FittedBlock.prototype.addDrawable.call( this, drawable );

      SVGGroup.addDrawable( this, drawable );
      drawable.updateSVGBlock( this );
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
      return 'SVGBlock#' + this.id + '-' + FittedBlock.fitString[ this.fit ];
    }
  } );

  Poolable.mixInto( SVGBlock, {
    initialize: SVGBlock.prototype.initialize
  } );

  return SVGBlock;
} );
