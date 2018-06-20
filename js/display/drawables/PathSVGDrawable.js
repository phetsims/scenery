// Copyright 2016, University of Colorado Boulder

/**
 * SVG drawable for Path nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var PathStatefulDrawable = require( 'SCENERY/display/drawables/PathStatefulDrawable' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Path. One of these drawables will be created
   * for each displayed instance of a Path.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function PathSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }

  scenery.register( 'PathSVGDrawable', PathSVGDrawable );

  inherit( SVGSelfDrawable, PathSVGDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable trait (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @public (scenery-internal)
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {PathSVGDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGPathElements ); // usesPaint: true

      // @protected {SVGPathElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
      this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'path' );

      return this;
    },

    /**
     * Updates the SVG elements so that they will appear like the current node's representation.
     * @protected
     *
     * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
     */
    updateSVGSelf: function() {
      assert && assert( !this.node.requiresSVGBoundsWorkaround(),
        'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );

      var path = this.svgElement;
      if ( this.dirtyShape ) {
        var svgPath = this.node.hasShape() ? this.node._shape.getSVGPath() : '';

        // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
        // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
        // the attribute can cause this bug
        if ( !svgPath ) { svgPath = 'M0 0'; }

        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      }

      // Apply any fill/stroke changes to our element.
      this.updateFillStrokeStyle( path );
    }
  } );

  PathStatefulDrawable.mixInto( PathSVGDrawable );

  Poolable.mixInto( PathSVGDrawable, {
    initialize: PathSVGDrawable.prototype.initialize
  } );

  return PathSVGDrawable;
} );
