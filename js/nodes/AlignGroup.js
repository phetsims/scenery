// Copyright 2016-2016, University of Colorado Boulder

/**
 * A group of alignment boxes that follow the constraints:
 * 1. Every box will have the same bounds, with an upper-left of (0,0)
 * 2. The box sizes will be the smallest possible to fit every box's content (with respective padding).
 * 3. Each box is responsible for positioning its content in its bounds (with customizable alignment and padding).
 *
 * Align boxes can be dynamically created and disposed, and only active boxes will be considered for the bounds.
 *
 * Since many sun components do not support resizing their contents dynamically, you may need to populate the AlignGroup
 * in the order of largest to smallest so that a fixed size container is large enough to contain the largest item.
 *
 * NOTE: Align box resizes may not happen immediately, and may be delayed until bounds of a align box's child occurs.
 *       layout updates can be forced with group.updateLayout(). If the align box's content that changed is connected
 *       to a Scenery display, its bounds will update when Display.updateDisplay() will called, so this will guarantee
 *       that the layout will be applied before it is displayed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var AlignBox = require( 'SCENERY/nodes/AlignBox' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Node = require( 'SCENERY/nodes/Node' );
  var NumberProperty = require( 'AXON/NumberProperty' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  /**
   * Creates an alignment group that can be composed of multiple boxes.
   * @constructor
   * @public
   *
   * Use createBox() to create alignment boxes. You can dispose() individual boxes, or call dispose() on this
   * group to dispose all of them.
   *
   * It is also possible to create AlignBox instances independently and assign their 'group' to this AlignGroup.
   *
   * @param {Object} [options]
   */
  function AlignGroup( options ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on options object is a code smell' );

    options = _.extend( {
      matchHorizontal: true, // {boolean} - Whether the boxes should have all matching widths (otherwise it fits to size)
      matchVertical: true  // {boolean} - Whether the boxes should have all matching heights (otherwise it fits to size)
    }, options );

    assert && assert( typeof options.matchHorizontal === 'boolean' );
    assert && assert( typeof options.matchVertical === 'boolean' );

    // @private {Array.<AlignBox>}
    this._alignBoxes = [];

    // @private {boolean}
    this._matchHorizontal = options.matchHorizontal;

    // @private {boolean}
    this._matchVertical = options.matchVertical;

    // @private {boolean} - Gets locked when certain layout is performed.
    this._resizeLock = false;

    // @private {Property.<boolean>}
    this._maxWidthProperty = new NumberProperty( 0 );

    // @private {Property.<boolean>}
    this._maxHeightProperty = new NumberProperty( 0 );

    // @private {number}
    this.id = globalId++;
  }

  scenery.register( 'AlignGroup', AlignGroup );

  inherit( Object, AlignGroup, {
    /**
     * Returns the current maximum width of the grouped content.
     * @public
     *
     * @returns {number} - Non-negative amount of width
     */
    getMaxWidth: function() {
      return this._maxWidthProperty.value;
    },
    get maxWidth() { return this.getMaxWidth(); },

    /**
     * Returns the Property holding the current maximum width of the grouped content.
     * @public
     *
     * @returns {Property.<number>} - Property with a non-negative amount of width
     */
    getMaxWidthProperty: function() {
      return this._maxWidthProperty;
    },
    get maxWidthProperty() { return this.getMaxWidthProperty(); },

    /**
     * Returns the current maximum height of the grouped content.
     * @public
     *
     * @returns {number} - Non-negative amount of height
     */
    getMaxHeight: function() {
      return this._maxHeightProperty.value;
    },
    get maxHeight() { return this.getMaxHeight(); },

    /**
     * Returns the Property holding the current maximum height of the grouped content.
     * @public
     *
     * @returns {Property.<number>} - Property with a non-negative amount of height
     */
    getMaxHeightProperty: function() {
      return this._maxHeightProperty;
    },
    get maxHeightProperty() { return this.getMaxHeightProperty(); },

    /**
     * Creates an alignment box with the given content and options.
     * @public
     *
     * @param {Node} content - Note that the content may be repositioned into place.
     * @param {Object} [options] - See AlignBox's constructor below for specific alignment options (it will be passed
     *                             through to the constructed AlignBox).
     */
    createBox: function( content, options ) {
      assert && assert( content instanceof Node );

      // Setting the group should call our addAlignBox()
      return new AlignBox( content, _.extend( {
        group: this
      }, options ) );
    },

    /**
     * Sets whether the widths of the align boxes should all match. If false, each box will use its preferred width
     * (usually equal to the content width + horizontal margins).
     * @public
     *
     * @param {boolean} matchHorizontal
     * @returns {AlignGroup} - For chaining
     */
    setMatchHorizontal: function( matchHorizontal ) {
      assert && assert( typeof matchHorizontal === 'boolean' );

      if ( this._matchHorizontal !== matchHorizontal ) {
        this._matchHorizontal = matchHorizontal;

        // Update layout, since it will probably change
        this.updateLayout();
      }

      return this;
    },
    set matchHorizontal( value ) { this.setMatchHorizontal( value ); },

    /**
     * Returns whether boxes currently are horizontally matched. See setMatchHorizontal() for details.
     * @public
     *
     * @returns {boolean}
     */
    getMatchHorizontal: function() {
      return this._matchHorizontal;
    },
    get matchHorizontal() { return this.getMatchHorizontal(); },

    /**
     * Sets whether the heights of the align boxes should all match. If false, each box will use its preferred height
     * (usually equal to the content height + vertical margins).
     * @public
     *
     * @param {boolean} matchVertical
     * @returns {AlignGroup} - For chaining
     */
    setMatchVertical: function( matchVertical ) {
      assert && assert( typeof matchVertical === 'boolean' );

      if ( this._matchVertical !== matchVertical ) {
        this._matchVertical = matchVertical;

        // Update layout, since it will probably change
        this.updateLayout();
      }

      return this;
    },
    set matchVertical( value ) { this.setMatchVertical( value ); },

    /**
     * Returns whether boxes currently are vertically matched. See setMatchVertical() for details.
     * @public
     *
     * @returns {boolean}
     */
    getMatchVertical: function() {
      return this._matchVertical;
    },
    get matchVertical() { return this.getMatchVertical(); },

    /**
     * Dispose all of the boxes.
     * @public
     */
    dispose: function() {
      for ( var i = this._alignBoxes.length - 1; i >= 0; i-- ) {
        this._alignBoxes[ i ].dispose();
      }
    },

    /**
     * Updates the localBounds and alignment for each alignBox.
     * @public
     *
     * NOTE: Calling this will usually not be necessary outside of Scenery, but this WILL trigger bounds revalidation
     *       for every alignBox, which can force the layout code to run.
     */
    updateLayout: function() {
      if ( this._resizeLock ) { return; }
      this._resizeLock = true;

      sceneryLog && sceneryLog.AlignGroup && sceneryLog.AlignGroup(
        'AlignGroup#' + this.id + ' updateLayout' );
      sceneryLog && sceneryLog.AlignGroup && sceneryLog.push();

      sceneryLog && sceneryLog.AlignGroup && sceneryLog.AlignGroup( 'AlignGroup computing maximum dimension' );
      sceneryLog && sceneryLog.AlignGroup && sceneryLog.push();

      // Compute the maximum dimension of our alignBoxs' content
      var maxWidth = 0;
      var maxHeight = 0;
      for ( var i = 0; i < this._alignBoxes.length; i++ ) {
        var alignBox = this._alignBoxes[ i ];

        var bounds = alignBox.getContentBounds();

        // Ignore bad bounds
        if ( bounds.isEmpty() || !bounds.isFinite() ) {
          continue;
        }

        maxWidth = Math.max( maxWidth, bounds.width );
        maxHeight = Math.max( maxHeight, bounds.height );
      }

      sceneryLog && sceneryLog.AlignGroup && sceneryLog.pop();
      sceneryLog && sceneryLog.AlignGroup && sceneryLog.AlignGroup( 'AlignGroup applying to boxes' );
      sceneryLog && sceneryLog.AlignGroup && sceneryLog.push();

      this._maxWidthProperty.value = maxWidth;
      this._maxHeightProperty.value = maxHeight;

      if ( maxWidth > 0 && maxHeight > 0 ) {
        // Apply that maximum dimension for each alignBox
        for ( i = 0; i < this._alignBoxes.length; i++ ) {
          this.setBoxBounds( this._alignBoxes[ i ], maxWidth, maxHeight );
        }
      }

      sceneryLog && sceneryLog.AlignGroup && sceneryLog.pop();
      sceneryLog && sceneryLog.AlignGroup && sceneryLog.pop();

      this._resizeLock = false;
    },

    /**
     * Sets a box's bounds based on our maximum dimensions.
     * @private
     *
     * @param {AlignBox} alignBox
     * @param {number} maxWidth
     * @param {number} maxHeight
     */
    setBoxBounds: function( alignBox, maxWidth, maxHeight ) {
      var alignBounds;

      // If we match both dimensions, we don't have to inspect the box's preferred size
      if ( this._matchVertical && this._matchHorizontal ) {
        alignBounds = new Bounds2( 0, 0, maxWidth, maxHeight );
      }
      else {
        // Grab the preferred size
        var contentBounds = alignBox.getContentBounds();

        // Match one orientation
        if ( this._matchVertical ) {
          alignBounds = new Bounds2( 0, 0, contentBounds.width, maxHeight );
        }
        else if ( this._matchHorizontal ) {
          alignBounds = new Bounds2( 0, 0, maxWidth, contentBounds.height );
        }
        // If not matching anything, just use its preferred size
        else {
          alignBounds = contentBounds;
        }
      }

      alignBox.alignBounds = alignBounds;
    },

    /**
     * Lets the group know that the alignBox has had its content resized. Called by the AlignBox
     * @public (scenery-internal)
     *
     * @param {AlignBox} alignBox
     */
    onAlignBoxResized: function( alignBox ) {
      // TODO: in the future, we could only update this specific alignBox if the others don't need updating.
      this.updateLayout();
    },

    /**
     * Adds the AlignBox to the group
     * @private
     *
     * @param {AlignBox} alignBox
     */
    addAlignBox: function( alignBox ) {
      this._alignBoxes.push( alignBox );

      // Trigger an update when a alignBox is added
      this.updateLayout();
    },

    /**
     * Removes the AlignBox from the group
     * @public (scenery-internal)
     *
     * @param {AlignBox} alignBox
     */
    removeAlignBox: function( alignBox ) {
      arrayRemove( this._alignBoxes, alignBox );

      // Trigger an update when a alignBox is removed
      this.updateLayout();
    }
  } );

  return AlignGroup;
} );
