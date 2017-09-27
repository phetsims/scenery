// Copyright 2017, University of Colorado Boulder

/**
 * A Node for a focus highlight that takes a shape and creates a Path with the default styling of a focus highlight
 * for a11y. The FocusHighlight has two paths.  The FocusHighlight path is an 'outer' highlight that is a little
 * lighter in color and transparency.  It as a child 'inner' path that is darker and more opaque, which gives the
 * focus highlight the illusion that it fades out.
 *
 * @author - Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var FocusOverlay = require( 'SCENERY/overlays/FocusOverlay' );
  var Path = require( 'SCENERY/nodes/Path' );

  /**
   * @constructor
   *
   * @param {Shape} shape - the shape for the focus highlight
   * @param {Object} options
   */
  function FocusHighlightPath( shape, options ) {

    Path.call( this, shape );

    options = _.extend( {
      outerStroke: FocusOverlay.focusColor,
      innerStroke: FocusOverlay.innerFocusColor,

      outerLineWidth: FocusOverlay.getFocusHighlightLineWidth( this ),
      innerLineWidth: FocusOverlay.getFocusHighlightInnerLineWidth( this )
    }, options );

    // options for this Path, the outer focus highlight
    var outerHighlightOptions = _.extend( {
      stroke: options.outerStroke,
      lineWidth: options.outerLineWidth
    }, options );
    this.mutate( outerHighlightOptions );

    // create the 'inner' focus highlight, the slightly darker and more opaque path that is on the inside
    // of the outer path to give the focus highlight a 'fade-out' appearance
    this.innerHighlightPath = new Path( shape, {
      stroke: options.innerStroke,
      lineWidth: options.innerLineWidth
    } );
    this.addChild( this.innerHighlightPath );
  }

  scenery.register( 'FocusHighlightPath', FocusHighlightPath );

  return inherit( Path, FocusHighlightPath, {

    /**
     * Update the shape of the child path (inner highlight) and this path (outer highlight).
     *
     * @param {Shape} shape
     */
    setHighlightShape: function( shape ) {
      this.setShape( shape );
      this.innerHighlightPath.setShape( shape );
    }
  } );
} );
