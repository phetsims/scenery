// Copyright 2016-2016, University of Colorado Boulder

/**
 * A container Node that will align content within a specific bounding box.
 *
 * If a custom alignBounds is provided, content will be aligned within that bounding box. Otherwise, it will be aligned
 * within a bounding box with the left-top corner of (0,0) of the necessary size to include both the content and
 * all of the margins.
 *
 * There are four margins: left, right, top, bottom. They can be set independently, or multiple can be set at the
 * same time (xMargin, yMargin and margin).
 *
 * NOTE: Container resize may not happen immediately, and may be delayed until bounds of a container's child occurs.
 *       layout updates can be forced with invalidateAlignment(). If the container's content that changed is connected
 *       to a Scenery display, its bounds will update when Display.updateDisplay() will called, so this will guarantee
 *       that the layout will be applied before it is displayed. container.getBounds() will not force a refresh, and
 *       may return stale bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  var ALIGNMENT_CONTAINER_OPTION_KEYS = [
    'alignBounds', // {Bounds2|null} - See setAlignBounds() for more documentation
    'xAlign', // {string} - 'left', 'center', or 'right', see setXAlign() for more documentation
    'yAlign', // {string} - 'top', 'center', or 'bottom', see setYAlign() for more documentation
    'margin', // {number} - Sets all margins, see setMargin() for more documentation
    'xMargin', // {number} - Sets horizontal margins, see setXMargin() for more documentation
    'yMargin', // {number} - Sets vertical margins, see setYMargin() for more documentation
    'leftMargin', // {number} - Sets left margin, see setLeftMargin() for more documentation
    'rightMargin', // {number} - Sets right margin, see setRightMargin() for more documentation
    'topMargin', // {number} - Sets top margin, see setTopMargin() for more documentation
    'bottomMargin', // {number} - Sets bottom margin, see setBottomMargin() for more documentation
    'group' // {AlignmentGroup|null} - Share bounds with others, see setGroup() for more documentation
  ];

  /**
   * An individual container for an alignment group. Will maintain its size to match that of the group by overriding
   * its localBounds, and will position its content inside its localBounds by respecting its alignment and margins.
   * @constructor
   * @public
   *
   * @param {Node} content - Content to align inside of the container
   * @param {Object} [options] - AlignmentContainer-specific options are documented in ALIGNMENT_CONTAINER_OPTION_KEYS
   *                             above, and can be provided along-side options for Node
   */
  function AlignmentContainer( content, options ) {

    // @private {Node} - Our actual content
    this._content = content;

    // @private {Bounds2|null} - Controls the bounds in which content is aligned.
    this._alignBounds = null;

    // @private {string} - How to align the content when the alignBounds are larger than our content with its margins.
    this._xAlign = 'center';
    this._yAlign = 'center';

    // @private {number} - How much space should be on each side.
    this._leftMargin = 0;
    this._rightMargin = 0;
    this._topMargin = 0;
    this._bottomMargin = 0;

    // @private {AlignmentGroup|null} - If available, an AlignmentGroup that will control our alignBounds
    this._group = null;

    // @private {function} - Callback for when bounds change (takes no arguments)
    this._contentBoundsListener = this.invalidateAlignment.bind( this );

    // Will be removed by dispose()
    this._content.on( 'bounds', this._contentBoundsListener );

    Node.call( this, _.extend( {}, options, {
      children: [ this._content ]
    } ) );
  }

  scenery.register( 'AlignmentContainer', AlignmentContainer );

  inherit( Node, AlignmentContainer, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: ALIGNMENT_CONTAINER_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * Triggers recomputation of the alignment. Should be called if it needs to be refreshed.
     * @public
     *
     * NOTE: container.getBounds() will not trigger a bounds validation for our content, and thus WILL NOT trigger
     * layout. content.getBounds() should trigger it, but invalidateAligment() is the preferred method for forcing a
     * re-check.
     */
    invalidateAlignment: function() {
      // The group update will change our alignBounds if required.
      if ( this._group ) {
        this._group.onContainerContentResized( this );
      }

      // If the alignBounds didn't change, we'll still need to update our own layout
      this.updateLayout();
    },

    /**
     * Sets the alignment bounds (the bounds in which our content will be aligned). If null, AlignmentContainer will act
     * as if the alignment bounds have a left-top corner of (0,0) and with a width/height that fits the content and
     * bounds.
     * @public
     *
     * NOTE: If the group is a valid AlignmentGroup, it will be responsible for setting the alignBounds.
     *
     * @param {Bounds2|null} alignBounds
     * @returns {AlignmentContainer} - For chaining
     */
    setAlignBounds: function( alignBounds ) {
      assert && assert( alignBounds === null || ( alignBounds instanceof Bounds2 && !alignBounds.isEmpty() && alignBounds.isFinite() ),
        'alignBounds should be a non-empty finite Bounds2' );

      // See if the bounds have changed. If both are Bounds2 with the same value, we won't update it.
      if ( this._alignBounds !== alignBounds &&
           ( !alignBounds ||
             !this._alignBounds ||
             !alignBounds.equals( this._alignBounds ) ) ) {
        this._alignBounds = alignBounds;

        this.updateLayout();
      }
      return this;
    },
    set alignBounds( value ) { this.setAlignBounds( value ); },

    /**
     * Returns the current alignment bounds (if available, see setAlignBounds for details).
     * @public
     *
     * @returns {Bounds2|null}
     */
    getAlignBounds: function() {
      return this._alignBounds;
    },
    get alignBounds() { return this.getAlignBounds(); },

    /**
     * Sets the attachment to an AlignmentGroup. When attached, our alignBounds will be controlled by the group.
     * @public
     *
     * @param {AlignmentGroup|null} group
     * @returns {AlignmentContainer} - For chaining
     */
    setGroup: function( group ) {
      assert && assert( group instanceof scenery.AlignmentGroup, 'group should be an AlignmentGroup' );

      if ( this._group !== group ) {
        // Remove from a previous group
        if ( this._group ) {
          this._group.removeContainer( this );
        }

        this._group = group;

        // Add to a new group
        if ( this._group ) {
          this._group.addContainer( this );
        }
      }

      return this;
    },
    set group( value ) { this.setGroup( value ); },

    /**
     * Returns the attached alignment group (if one exists), or null otherwise.
     * @public
     *
     * @returns {AlignmentGroup|null}
     */
    getGroup: function() {
      return this._group;
    },
    get group() { return this.getGroup(); },

    /**
     * Sets the horizontal alignment of this container.
     * @public
     *
     * Available values are 'left', 'center', or 'right'.
     *
     * @param {string} xAlign
     * @returns {AlignmentContainer} - For chaining
     */
    setXAlign: function( xAlign ) {
      assert && assert( xAlign === 'left' || xAlign === 'center' || xAlign === 'right',
        'xAlign should be one of: \'left\', \'center\', or \'right\'' );

      if ( this._xAlign !== xAlign ) {
        this._xAlign = xAlign;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set xAlign( value ) { this.setXAlign( value ); },

    /**
     * Returns the current horizontal alignment of this container.
     * @public
     *
     * @returns {string} - See setXAlign for values.
     */
    getXAlign: function() {
      return this._xAlign;
    },
    get xAlign() { return this.getXAlign(); },

    /**
     * Sets the vertical alignment of this container.
     * @public
     *
     * Available values are 'top', 'center', or 'bottom'.
     *
     * @param {string} yAlign
     * @returns {AlignmentContainer} - For chaining
     */
    setYAlign: function( yAlign ) {
      assert && assert( yAlign === 'top' || yAlign === 'center' || yAlign === 'bottom',
        'yAlign should be one of: \'top\', \'center\', or \'bottom\'' );

      if ( this._yAlign !== yAlign ) {
        this._yAlign = yAlign;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set yAlign( value ) { this.setYAlign( value ); },

    /**
     * Returns the current vertical alignment of this container.
     * @public
     *
     * @returns {string} - See setYAlign for values.
     */
    getYAlign: function() {
      return this._yAlign;
    },
    get yAlign() { return this.getYAlign(); },

    /**
     * Sets the margin of this container (setting margin values for all sides at once).
     * @public
     *
     * This margin is the minimum amount of horizontal space that will exist between the content the sides of this
     * container.
     *
     * @param {number} margin
     * @returns {AlignmentContainer} - For chaining
     */
    setMargin: function( margin ) {
      assert && assert( typeof margin === 'number' && isFinite( margin ) && margin >= 0,
        'margin should be a finite non-negative number' );

      if ( this._leftMargin !== margin ||
           this._rightMargin !== margin ||
           this._topMargin !== margin ||
           this._bottomMargin !== margin ) {
        this._leftMargin = this._rightMargin = this._topMargin = this._bottomMargin = margin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set margin( value ) { this.setMargin( value ); },

    /**
     * Returns the current margin of this container (assuming all margin values are the same).
     * @public
     *
     * @returns {number} - See setMargin for more information.
     */
    getMargin: function() {
      assert && assert( this._leftMargin === this._rightMargin &&
                        this._leftMargin === this._topMargin &&
                        this._leftMargin === this._bottomMargin,
        'Getting margin does not have a unique result if the left and right margins are different' );
      return this._leftMargin;
    },
    get margin() { return this.getMargin(); },

    /**
     * Sets the horizontal margin of this container (setting both left and right margins at once).
     * @public
     *
     * This margin is the minimum amount of horizontal space that will exist between the content and the left and
     * right sides of this container.
     *
     * @param {number} xMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setXMargin: function( xMargin ) {
      assert && assert( typeof xMargin === 'number' && isFinite( xMargin ) && xMargin >= 0,
        'xMargin should be a finite non-negative number' );

      if ( this._leftMargin !== xMargin || this._rightMargin !== xMargin ) {
        this._leftMargin = this._rightMargin = xMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set xMargin( value ) { this.setXMargin( value ); },

    /**
     * Returns the current horizontal margin of this container (assuming the left and right margins are the same).
     * @public
     *
     * @returns {number} - See setXMargin for more information.
     */
    getXMargin: function() {
      assert && assert( this._leftMargin === this._rightMargin,
        'Getting xMargin does not have a unique result if the left and right margins are different' );
      return this._leftMargin;
    },
    get xMargin() { return this.getXMargin(); },

    /**
     * Sets the vertical margin of this container (setting both top and bottom margins at once).
     * @public
     *
     * This margin is the minimum amount of vertical space that will exist between the content and the top and
     * bottom sides of this container.
     *
     * @param {number} yMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setYMargin: function( yMargin ) {
      assert && assert( typeof yMargin === 'number' && isFinite( yMargin ) && yMargin >= 0,
        'yMargin should be a finite non-negative number' );

      if ( this._topMargin !== yMargin || this._bottomMargin !== yMargin ) {
        this._topMargin = this._bottomMargin = yMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set yMargin( value ) { this.setYMargin( value ); },

    /**
     * Returns the current vertical margin of this container (assuming the top and bottom margins are the same).
     * @public
     *
     * @returns {number} - See setYMargin for more information.
     */
    getYMargin: function() {
      assert && assert( this._topMargin === this._bottomMargin,
        'Getting yMargin does not have a unique result if the top and bottom margins are different' );
      return this._topMargin;
    },
    get yMargin() { return this.getYMargin(); },

    /**
     * Sets the left margin of this container.
     * @public
     *
     * This margin is the minimum amount of horizontal space that will exist between the content and the left side of
     * the container.
     *
     * @param {number} leftMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setLeftMargin: function( leftMargin ) {
      assert && assert( typeof leftMargin === 'number' && isFinite( leftMargin ) && leftMargin >= 0,
        'leftMargin should be a finite non-negative number' );

      if ( this._leftMargin !== leftMargin ) {
        this._leftMargin = leftMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set leftMargin( value ) { this.setLeftMargin( value ); },

    /**
     * Returns the current left margin of this container.
     * @public
     *
     * @returns {number} - See setLeftMargin for more information.
     */
    getLeftMargin: function() {
      return this._leftMargin;
    },
    get leftMargin() { return this.getLeftMargin(); },

    /**
     * Sets the right margin of this container.
     * @public
     *
     * This margin is the minimum amount of horizontal space that will exist between the content and the right side of
     * the container.
     *
     * @param {number} rightMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setRightMargin: function( rightMargin ) {
      assert && assert( typeof rightMargin === 'number' && isFinite( rightMargin ) && rightMargin >= 0,
        'rightMargin should be a finite non-negative number' );

      if ( this._rightMargin !== rightMargin ) {
        this._rightMargin = rightMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set rightMargin( value ) { this.setRightMargin( value ); },

    /**
     * Returns the current right margin of this container.
     * @public
     *
     * @returns {number} - See setRightMargin for more information.
     */
    getRightMargin: function() {
      return this._rightMargin;
    },
    get rightMargin() { return this.getRightMargin(); },

    /**
     * Sets the top margin of this container.
     * @public
     *
     * This margin is the minimum amount of vertical space that will exist between the content and the top side of the
     * container.
     *
     * @param {number} topMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setTopMargin: function( topMargin ) {
      assert && assert( typeof topMargin === 'number' && isFinite( topMargin ) && topMargin >= 0,
        'topMargin should be a finite non-negative number' );

      if ( this._topMargin !== topMargin ) {
        this._topMargin = topMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set topMargin( value ) { this.setTopMargin( value ); },

    /**
     * Returns the current top margin of this container.
     * @public
     *
     * @returns {number} - See setTopMargin for more information.
     */
    getTopMargin: function() {
      return this._topMargin;
    },
    get topMargin() { return this.getTopMargin(); },

    /**
     * Sets the bottom margin of this container.
     * @public
     *
     * This margin is the minimum amount of vertical space that will exist between the content and the bottom side of the
     * container.
     *
     * @param {number} bottomMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setBottomMargin: function( bottomMargin ) {
      assert && assert( typeof bottomMargin === 'number' && isFinite( bottomMargin ) && bottomMargin >= 0,
        'bottomMargin should be a finite non-negative number' );

      if ( this._bottomMargin !== bottomMargin ) {
        this._bottomMargin = bottomMargin;

        // Trigger re-layout
        this.invalidateAlignment();
      }

      return this;
    },
    set bottomMargin( value ) { this.setBottomMargin( value ); },

    /**
     * Returns the current bottom margin of this container.
     * @public
     *
     * @returns {number} - See setBottomMargin for more information.
     */
    getBottomMargin: function() {
      return this._bottomMargin;
    },
    get bottomMargin() { return this.getBottomMargin(); },

    /**
     * Returns the bounding box of this container's content. This will include any margins.
     * @private
     *
     * @returns {Bounds2}
     */
    getContentBounds: function() {
      var bounds = this._content.bounds;
      return new Bounds2( bounds.left - this._leftMargin,
                          bounds.top - this._topMargin,
                          bounds.right + this._rightMargin,
                          bounds.bottom + this._bottomMargin );
    },

    /**
     * Updates the layout of this alignment container.
     * @private
     */
    updateLayout: function() {
      // If we have alignBounds, use that.
      if ( this._alignBounds !== null ) {
        this.localBounds = this._alignBounds;
      }
      // Otherwise, we'll grab a Bounds2 anchored at the upper-left with our required dimensions.
      else {
        var widthWithMargin = this._leftMargin + this._content.width + this._rightMargin;
        var heightWithMargin = this._topMargin + this._content.height + this._bottomMargin;
        this.localBounds = new Bounds2( 0, 0, widthWithMargin, heightWithMargin );
      }

      if ( this._xAlign === 'center' ) {
        this._content.centerX = this.localBounds.centerX;
      }
      else if ( this._xAlign === 'left' ) {
        this._content.left = this.localBounds.left + this._leftMargin;
      }
      else if ( this._xAlign === 'right' ) {
        this._content.right = this.localBounds.right - this._rightMargin;
      }
      else {
        assert && assert( 'Bad xAlign: ' + this._xAlign );
      }

      if ( this._yAlign === 'center' ) {
        this._content.centerY = this.localBounds.centerY;
      }
      else if ( this._yAlign === 'top' ) {
        this._content.top = this.localBounds.top + this._topMargin;
      }
      else if ( this._yAlign === 'bottom' ) {
        this._content.bottom = this.localBounds.bottom - this._bottomMargin;
      }
      else {
        assert && assert( 'Bad yAlign: ' + this._yAlign );
      }

      assert && assert( this.localBounds.dilated( 1e-5 ).containsBounds( this._content.bounds ),
        'All of our contents should be contained in our localBounds' );
    },

    /**
     * Disposes this container, so that the alignment group won't update this container, and won't use its bounds for
     * laying out the other containers.
     * @public
     */
    dispose: function() {
      this._content.off( 'bounds', this._contentBoundsListener );

      if ( this._group ) {
        this._group.removeContainer( this );
      }
    }
  } );

  return AlignmentContainer;
} );
