// Copyright 2016, University of Colorado Boulder

/**
 * SVG drawable for Rectangle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var ExperimentalPoolable = require( 'PHET_CORE/ExperimentalPoolable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var RectangleStatefulDrawable = require( 'SCENERY/display/drawables/RectangleStatefulDrawable' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGRectangleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function RectangleSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }

  scenery.register( 'RectangleSVGDrawable', RectangleSVGDrawable );

  inherit( SVGSelfDrawable, RectangleSVGDrawable, {
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
     * @returns {RectangleSVGDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGRectangleElements ); // usesPaint: true

      this.lastArcW = -1; // invalid on purpose
      this.lastArcH = -1; // invalid on purpose

      // @protected {SVGRectElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
      this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'rect' );

      return this;
    },

    /**
     * Updates the SVG elements so that they will appear like the current node's representation.
     * @protected
     *
     * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
     */
    updateSVGSelf: function() {
      var rect = this.svgElement;

      if ( this.dirtyX ) {
        rect.setAttribute( 'x', this.node._rectX );
      }
      if ( this.dirtyY ) {
        rect.setAttribute( 'y', this.node._rectY );
      }
      if ( this.dirtyWidth ) {
        rect.setAttribute( 'width', this.node._rectWidth );
      }
      if ( this.dirtyHeight ) {
        rect.setAttribute( 'height', this.node._rectHeight );
      }
      if ( this.dirtyCornerXRadius || this.dirtyCornerYRadius || this.dirtyWidth || this.dirtyHeight ) {
        var arcw = 0;
        var arch = 0;

        // workaround for various browsers if rx=20, ry=0 (behavior is inconsistent, either identical to rx=20,ry=20, rx=0,ry=0. We'll treat it as rx=0,ry=0)
        // see https://github.com/phetsims/scenery/issues/183
        if ( this.node.isRounded() ) {
          var maximumArcSize = this.node.getMaximumArcSize();
          arcw = Math.min( this.node._cornerXRadius, maximumArcSize );
          arch = Math.min( this.node._cornerYRadius, maximumArcSize );
        }
        if ( arcw !== this.lastArcW ) {
          this.lastArcW = arcw;
          rect.setAttribute( 'rx', arcw );
        }
        if ( arch !== this.lastArcH ) {
          this.lastArcH = arch;
          rect.setAttribute( 'ry', arch );
        }
      }

      // Apply any fill/stroke changes to our element.
      this.updateFillStrokeStyle( rect );
    }
  } );

  RectangleStatefulDrawable.mixInto( RectangleSVGDrawable );

  ExperimentalPoolable.mixInto( RectangleSVGDrawable, {
    initialize: RectangleSVGDrawable.prototype.initialize
  } );

  return RectangleSVGDrawable;
} );
