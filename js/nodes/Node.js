// Copyright 2012-2020, University of Colorado Boulder

/**
 * A Node for the Scenery scene graph. Supports general directed acyclic graphics (DAGs).
 * Handles multiple layers with assorted types (Canvas 2D, SVG, DOM, WebGL, etc.).
 *
 * ## General description of Nodes
 *
 * In Scenery, the visual output is determined by a group of connected Nodes (generally known as a scene graph).
 * Each Node has a list of 'child' Nodes. When a Node is visually displayed, its child Nodes (children) will also be
 * displayed, along with their children, etc. There is typically one 'root' Node that is passed to the Scenery Display
 * whose descendants (Nodes that can be traced from the root by child relationships) will be displayed.
 *
 * For instance, say there are Nodes named A, B, C, D and E, who have the relationships:
 * - B is a child of A (thus A is a parent of B)
 * - C is a child of A (thus A is a parent of C)
 * - D is a child of C (thus C is a parent of D)
 * - E is a child of C (thus C is a parent of E)
 * where A would be the root Node. This can be visually represented as a scene graph, where a line connects a parent
 * Node to a child Node (where the parent is usually always at the top of the line, and the child is at the bottom):
 * For example:
 *
 *   A
 *  / \
 * B   C
 *    / \
 *   D   E
 *
 * Additionally, in this case:
 * - D is a 'descendant' of A (due to the C being a child of A, and D being a child of C)
 * - A is an 'ancestor' of D (due to the reverse)
 * - C's 'subtree' is C, D and E, which consists of C itself and all of its descendants.
 *
 * Note that Scenery allows some more complicated forms, where Nodes can have multiple parents, e.g.:
 *
 *   A
 *  / \
 * B   C
 *  \ /
 *   D
 *
 * In this case, D has two parents (B and C). Scenery disallows any Node from being its own ancestor or descendant,
 * so that loops are not possible. When a Node has two or more parents, it means that the Node's subtree will typically
 * be displayed twice on the screen. In the above case, D would appear both at B's position and C's position. Each
 * place a Node would be displayed is known as an 'instance'.
 *
 * Each Node has a 'transform' associated with it, which determines how its subtree (that Node and all of its
 * descendants) will be positioned. Transforms can contain:
 * - Translation, which moves the position the subtree is displayed
 * - Scale, which makes the displayed subtree larger or smaller
 * - Rotation, which displays the subtree at an angle
 * - or any combination of the above that uses an affine matrix (more advanced transforms with shear and combinations
 *   are possible).
 *
 * Say we have the following scene graph:
 *
 *   A
 *   |
 *   B
 *   |
 *   C
 *
 * where there are the following transforms:
 * - A has a 'translation' that moves the content 100 pixels to the right
 * - B has a 'scale' that doubles the size of the content
 * - C has a 'rotation' that rotates 180-degrees around the origin
 *
 * If C displays a square that fills the area with 0 <= x <= 10 and 0 <= y <= 10, we can determine the position on
 * the display by applying transforms starting at C and moving towards the root Node (in this case, A):
 * 1. We apply C's rotation to our square, so the filled area will now be -10 <= x <= 0 and -10 <= y <= 0
 * 2. We apply B's scale to our square, so now we have -20 <= x <= 0 and -20 <= y <= 0
 * 3. We apply A's translation to our square, moving it to 80 <= x <= 100 and -20 <= y <= 0
 *
 * Nodes also have a large number of properties that will affect how their entire subtree is rendered, such as
 * visibility, opacity, etc.
 *
 * ## Creating Nodes
 *
 * Generally, there are two types of Nodes:
 * - Nodes that don't display anything, but serve as a container for other Nodes (e.g. Node itself, HBox, VBox)
 * - Nodes that display content, but ALSO serve as a container (e.g. Circle, Image, Text)
 *
 * When a Node is created with the default Node constructor, e.g.:
 *   var node = new Node();
 * then that Node will not display anything by itself.
 *
 * Generally subtypes of Node are used for displaying things, such as Circle, e.g.:
 *   var circle = new Circle( 20 ); // radius of 20
 *
 * Almost all Nodes (with the exception of leaf-only Nodes like Spacer) can contain children.
 *
 * ## Connecting Nodes, and rendering order
 *
 * To make a 'childNode' become a 'parentNode', the typical way is to call addChild():
 *   parentNode.addChild( childNode );
 *
 * To remove this connection, you can call:
 *   parentNode.removeChild( childNode );
 *
 * Adding a child Node with addChild() puts it at the end of parentNode's list of child Nodes. This is important,
 * because the order of children affects what Nodes are drawn on the 'top' or 'bottom' visually. Nodes that are at the
 * end of the list of children are generally drawn on top.
 *
 * This is generally easiest to represent by notating scene graphs with children in order from left to right, thus:
 *
 *   A
 *  / \
 * B   C
 *    / \
 *   D   E
 *
 * would indicate that A's children are [B,C], so C's subtree is drawn ON TOP of B. The same is true of C's children
 * [D,E], so E is drawn on top of D. If a Node itself has content, it is drawn below that of its children (so C itself
 * would be below D and E).
 *
 * This means that for every scene graph, Nodes instances can be ordered from bottom to top. For the above example, the
 * order is:
 * 1. A (on the very bottom visually, may get covered up by other Nodes)
 * 2. B
 * 3. C
 * 4. D
 * 5. E (on the very top visually, may be covering other Nodes)
 *
 * ## Trails
 *
 * For examples where there are multiple parents for some Nodes (also referred to as DAG in some code, as it represents
 * a Directed Acyclic Graph), we need more information about the rendering order (as otherwise Nodes could appear
 * multiple places in the visual bottom-to-top order.
 *
 * A Trail is basically a list of Nodes, where every Node in the list is a child of its previous element, and a parent
 * of its next element. Thus for the scene graph:
 *
 *   A
 *  / \
 * B   C
 *  \ / \
 *   D   E
 *    \ /
 *     F
 *
 * there are actually three instances of F being displayed, with three trails:
 * - [A,B,D,F]
 * - [A,C,D,F]
 * - [A,C,E,F]
 * Note that the trails are essentially listing Nodes used in walking from the root (A) to the relevant Node (F) using
 * connections between parents and children.
 *
 * The trails above are in order from bottom to top (visually), due to the order of children. Thus since A's children
 * are [B,C] in that order, F with the trail [A,B,D,F] is displayed below [A,C,D,F], because C is after B.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Property from '../../../axon/js/Property.js';
import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import TinyForwardingProperty from '../../../axon/js/TinyForwardingProperty.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import TinyStaticProperty from '../../../axon/js/TinyStaticProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Utils from '../../../dot/js/Utils.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Shape from '../../../kite/js/Shape.js';
import arrayDifference from '../../../phet-core/js/arrayDifference.js';
import deprecationWarning from '../../../phet-core/js/deprecationWarning.js';
import inherit from '../../../phet-core/js/inherit.js';
import merge from '../../../phet-core/js/merge.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import ParallelDOM from '../accessibility/pdom/ParallelDOM.js';
import Instance from '../display/Instance.js';
import Renderer from '../display/Renderer.js';
import Mouse from '../input/Mouse.js';
import Pen from '../input/Pen.js';
import Touch from '../input/Touch.js';
import scenery from '../scenery.js';
import CanvasContextWrapper from '../util/CanvasContextWrapper.js';
import Filter from '../util/Filter.js';
import Picker from '../util/Picker.js';
import RendererSummary from '../util/RendererSummary.js';
import Trail from '../util/Trail.js';

// constants
const clamp = Utils.clamp;

let globalIdCounter = 1;

const scratchBounds2 = Bounds2.NOTHING.copy(); // mutable {Bounds2} used temporarily in methods
const scratchMatrix3 = new Matrix3();

const PICKABLE_PROPERTY_TANDEM_NAME = 'pickableProperty';
const VISIBLE_PROPERTY_TANDEM_NAME = 'visibleProperty';

// Node options, in the order they are executed in the constructor/mutate()
const NODE_OPTION_KEYS = [
  'children', // {Array.<Node>}- List of children to add (in order), see setChildren for more documentation
  'cursor', // {string|null} - CSS cursor to display when over this Node, see setCursor() for more documentation

  'visibleProperty', // {Property.<boolean>|null} - Sets forwarding of the visibleProperty, see setVisibleProperty() for more documentation
  'visible', // {boolean} - Whether the Node is visible, see setVisible() for more documentation

  'pickablePropertyPhetioInstrumented', // {boolean} - When true, create an instrumented pickableProperty when this Node is instrumented, see setPickablePropertyPhetioInstrumented() for more documentation
  'pickableProperty', // {Property.<boolean|null>|null} - Sets forwarding of the pickableProperty, see setPickableProperty() for more documentation
  'pickable', // {boolean|null} - Whether the Node is pickable, see setPickable() for more documentation
  'inputEnabled', // {boolean} Whether input events can reach into this subtree, see setInputEnabled() for more documentation
  'inputListeners', // {Array.<Object>} - The input listeners attached to the Node, see setInputListeners() for more documentation
  'opacity', // {number} - Opacity of this Node's subtree, see setOpacity() for more documentation
  'filters', // {Array.<Filter>} - Non-opacity filters, see setFilters() for more documentation
  'matrix', // {Matrix3} - Transformation matrix of the Node, see setMatrix() for more documentation
  'translation', // {Vector2} - x/y translation of the Node, see setTranslation() for more documentation
  'x', // {number} - x translation of the Node, see setX() for more documentation
  'y', // {number} - y translation of the Node, see setY() for more documentation
  'rotation', // {number} - rotation (in radians) of the Node, see setRotation() for more documentation
  'scale', // {number} - scale of the Node, see scale() for more documentation
  'excludeInvisibleChildrenFromBounds', // {boolean} - Controls bounds depending on child visibility, see setExcludeInvisibleChildrenFromBounds() for more documentation
  'localBounds', // {Bounds2|null} - bounds of subtree in local coordinate frame, see setLocalBounds() for more documentation
  'maxWidth', // {number|null} - Constrains width of this Node, see setMaxWidth() for more documentation
  'maxHeight', // {number|null} - Constrains height of this Node, see setMaxHeight() for more documentation
  'leftTop', // {Vector2} - The upper-left corner of this Node's bounds, see setLeftTop() for more documentation
  'centerTop', // {Vector2} - The top-center of this Node's bounds, see setCenterTop() for more documentation
  'rightTop', // {Vector2} - The upper-right corner of this Node's bounds, see setRightTop() for more documentation
  'leftCenter', // {Vector2} - The left-center of this Node's bounds, see setLeftCenter() for more documentation
  'center', // {Vector2} - The center of this Node's bounds, see setCenter() for more documentation
  'rightCenter', // {Vector2} - The center-right of this Node's bounds, see setRightCenter() for more documentation
  'leftBottom', // {Vector2} - The bottom-left of this Node's bounds, see setLeftBottom() for more documentation
  'centerBottom', // {Vector2} - The middle center of this Node's bounds, see setCenterBottom() for more documentation
  'rightBottom', // {Vector2} - The bottom right of this Node's bounds, see setRightBottom() for more documentation
  'left', // {number} - The left side of this Node's bounds, see setLeft() for more documentation
  'right', // {number} - The right side of this Node's bounds, see setRight() for more documentation
  'top', // {number} - The top side of this Node's bounds, see setTop() for more documentation
  'bottom', // {number} - The bottom side of this Node's bounds, see setBottom() for more documentation
  'centerX', // {number} - The x-center of this Node's bounds, see setCenterX() for more documentation
  'centerY', // {number} - The y-center of this Node's bounds, see setCenterY() for more documentation
  'renderer', // {string|null} - The preferred renderer for this subtree, see setRenderer() for more documentation
  'layerSplit', // {boolean} - Forces this subtree into a layer of its own, see setLayerSplit() for more documentation
  'usesOpacity', // {boolean} - Hint that opacity will be changed, see setUsesOpacity() for more documentation
  'cssTransform', // {boolean} - Hint that can trigger using CSS transforms, see setCssTransform() for more documentation
  'excludeInvisible', // {boolean} - If this is invisible, exclude from DOM, see setExcludeInvisible() for more documentation
  'webglScale', // {number|null} - Hint to adjust WebGL scaling quality for this subtree, see setWebglScale() for more documentation
  'preventFit', // {boolean} - Prevents layers from fitting this subtree, see setPreventFit() for more documentation
  'mouseArea', // {Bounds2|Shape|null} - Changes the area the mouse can interact with, see setMouseArea() for more documentation
  'touchArea', // {Bounds2|Shape|null} - Changes the area touches can interact with, see setTouchArea() for more documentation
  'clipArea', // {Shape|null} - Makes things outside of a shape invisible, see setClipArea() for more documentation
  'transformBounds' // {boolean} - Flag that makes bounds tighter, see setTransformBounds() for more documentation
];

const DEFAULT_OPTIONS = {
  visible: true,
  opacity: 1,
  pickable: null,
  pickablePropertyPhetioInstrumented: false,
  inputEnabled: true,
  clipArea: null,
  mouseArea: null,
  touchArea: null,
  cursor: null,
  transformBounds: false,
  maxWidth: null,
  maxHeight: null,
  renderer: null,
  usesOpacity: false,
  layerSplit: false,
  cssTransform: false,
  excludeInvisible: false,
  webglScale: null,
  preventFit: false
};

/**
 * Creates a Node with options.
 * @public
 * @constructor
 * @mixes ParallelDOM
 *
 * NOTE: Directly created Nodes (not of any subtype, but created with "new Node( ... )") are generally used as
 *       containers, which can hold other Nodes, subtypes of Node that can display things.
 *
 * Node and its subtypes generally have the last constructor parameter reserved for the 'options' object. This is a
 * key-value map that specifies relevant options that are used by Node and subtypes.
 *
 * For example, one of Node's options is bottom, and one of Circle's options is radius. When a circle is created:
 *   var circle = new Circle( {
 *     radius: 10,
 *     bottom: 200
 *   } );
 * This will create a Circle, set its radius (by executing circle.radius = 10, which uses circle.setRadius()), and
 * then will align the bottom of the circle along y=200 (by executing circle.bottom = 200, which uses
 * node.setBottom()).
 *
 * The options are executed in the order specified by each types _mutatorKeys property.
 *
 * The options object is currently not checked to see whether there are property (key) names that are not used, so it
 * is currently legal to do "new Node( { fork_kitchen_spoon: 5 } )".
 *
 * Usually, an option (e.g. 'visible'), when used in a constructor or mutate() call, will directly use the ES5 setter
 * for that property (e.g. node.visible = ...), which generally forwards to a non-ES5 setter function
 * (e.g. node.setVisible( ... )) that is responsible for the behavior. Documentation is generally on these methods
 * (e.g. setVisible), although some methods may be dynamically created to avoid verbosity (like node.leftTop).
 *
 * Sometimes, options invoke a function instead (e.g. 'scale') because the verb and noun are identical. In this case,
 * instead of setting the setter (node.scale = ..., which would override the function), it will instead call
 * the method directly (e.g. node.scale( ... )).
 *
 * @param {Object} [options] - Optional options object, as described above.
 */
function Node( options ) {
  // NOTE: All member properties with names starting with '_' are assumed to be @private!

  // @private {number} - Assigns a unique ID to this Node (allows trails to get a unique list of IDs)
  this._id = globalIdCounter++;

  // @protected {Array.<Instance>} - All of the Instances tracking this Node
  this._instances = [];

  // @protected {Array.<Display>} - All displays where this Node is the root.
  this._rootedDisplays = [];

  // @protected {Array.<Drawable>} - Drawable states that need to be updated on mutations. Generally added by SVG and
  // DOM elements that need to closely track state (possibly by Canvas to maintain dirty state).
  this._drawables = [];

  // @private {TinyForwardingProperty.<boolean>} - Whether this Node (and its children) will be visible when the scene is updated.
  // Visible Nodes by default will not be pickable either.
  // NOTE: This is fired synchronously when the visibility of the Node is toggled
  this._visibleProperty = new TinyForwardingProperty( DEFAULT_OPTIONS.visible, true );
  this._visibleProperty.lazyLink( this.onVisiblePropertyChange.bind( this ) );

  // @public {TinyProperty.<number>} - Opacity, in the range from 0 (fully transparent) to 1 (fully opaque).
  // NOTE: This is fired synchronously when the opacity of the Node is toggled
  this.opacityProperty = new TinyProperty( DEFAULT_OPTIONS.opacity );
  this.opacityProperty.lazyLink( this.onOpacityPropertyChange.bind( this ) );

  // @private {TinyForwardingProperty.<boolean|null>} - See setPickable() and setPickableProperty()
  // NOTE: This is fired synchronously when the pickability of the Node is toggled
  this._pickableProperty = new TinyForwardingProperty( DEFAULT_OPTIONS.pickable, DEFAULT_OPTIONS.pickablePropertyPhetioInstrumented );
  this._pickableProperty.lazyLink( this.onPickablePropertyChange.bind( this ) );

  // @public {TinyProperty.<boolean>} - Whether input event listeners on this Node or descendants on a trail will have
  // input listeners. triggered. Note that this does NOT effect picking, and only prevents some listeners from being
  // fired.
  this.inputEnabledProperty = new TinyProperty( DEFAULT_OPTIONS.inputEnabled );

  // @private {TinyProperty.<Shape|null>} - This Node and all children will be clipped by this shape (in addition to any
  // other clipping shapes). The shape should be in the local coordinate frame.
  // NOTE: This is fired synchronously when the clipArea of the Node is toggled
  this.clipAreaProperty = new TinyProperty( DEFAULT_OPTIONS.clipArea );

  // @private - Areas for hit intersection. If set on a Node, no descendants can handle events.
  this._mouseArea = DEFAULT_OPTIONS.mouseArea; // {Shape|Bounds2} for mouse position in the local coordinate frame
  this._touchArea = DEFAULT_OPTIONS.touchArea; // {Shape|Bounds2} for touch and pen position in the local coordinate frame

  // @private {string|null} - The CSS cursor to be displayed over this Node. null should be the default (inherit) value.
  this._cursor = DEFAULT_OPTIONS.cursor;

  // @public (scenery-internal) - Not for public use, but used directly internally for performance.
  this._children = []; // {Array.<Node>} - Ordered array of child Nodes.
  this._parents = []; // {Array.<Node>} - Unordered array of parent Nodes.

  // @private {boolean} - Whether we will do more accurate (and tight) bounds computations for rotations and shears.
  this._transformBounds = DEFAULT_OPTIONS.transformBounds;

  /*
   * Set up the transform reference. we add a listener so that the transform itself can be modified directly
   * by reference, triggering the event notifications for Scenery The reference to the Transform3 will never change.
   */
  this._transform = new Transform3(); // @private {Transform3}
  this._transformListener = this.onTransformChange.bind( this ); // @private {Function}
  this._transform.changeEmitter.addListener( this._transformListener ); // NOTE: Listener/transform bound to this Node.

  /*
   * Maximum dimensions for the Node's local bounds before a corrective scaling factor is applied to maintain size.
   * The maximum dimensions are always compared to local bounds, and applied "before" the Node's transform.
   * Whenever the local bounds or maximum dimensions of this Node change and it has at least one maximum dimension
   * (width or height), an ideal scale is computed (either the smallest scale for our local bounds to fit the
   * dimension constraints, OR 1, whichever is lower). Then the Node's transform will be scaled (prepended) with
   * a scale adjustment of ( idealScale / alreadyAppliedScaleFactor ).
   * In the simple case where the Node isn't otherwise transformed, this will apply and update the Node's scale so that
   * the Node matches the maximum dimensions, while never scaling over 1. Note that manually applying transforms to
   * the Node is fine, but may make the Node's width greater than the maximum width.
   * NOTE: If a dimension constraint is null, no resizing will occur due to it. If both maxWidth and maxHeight are null,
   * no scale adjustment will be applied.
   *
   * Also note that setting maxWidth/maxHeight is like adding a local bounds listener (will trigger validation of
   * bounds during the updateDisplay step). NOTE: this means updates to the transform (on a local bounds change) will
   * happen when bounds are validated (validateBounds()), which does not happen synchronously on a child's size
   * change. It does happen at least once in updateDisplay() before rendering, and calling validateBounds() can force
   * a re-check and transform.
   */
  this._maxWidth = DEFAULT_OPTIONS.maxWidth; // @private {number|null}
  this._maxHeight = DEFAULT_OPTIONS.maxHeight; // @private {number|null}
  this._appliedScaleFactor = 1; // @private {number} - Scale applied due to the maximum dimension constraints.

  // @private {Array.<Function>} - For user input handling (mouse/touch).
  this._inputListeners = [];

  // Add listener count change notifications into these Properties, since we need to know when their number of listeners
  // changes dynamically.
  const boundsListenersAddedOrRemovedListener = this.onBoundsListenersAddedOrRemoved.bind( this );

  const boundsInvalidationListener = this.validateBounds.bind( this );
  const selfBoundsInvalidationListener = this.validateSelfBounds.bind( this );

  // @public {TinyStaticProperty.<Bounds2>} - [mutable] Bounds for this Node and its children in the "parent" coordinate
  // frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This is fired **asynchronously** (usually as part of a Display.updateDisplay()) when the bounds of the Node
  // is changed.
  this.boundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), boundsInvalidationListener );
  this.boundsProperty.changeCount = boundsListenersAddedOrRemovedListener;

  // @public {TinyStaticProperty.<Bounds2>} - [mutable] Bounds for this Node and its children in the "local" coordinate
  // frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This is fired **asynchronously** (usually as part of a Display.updateDisplay()) when the localBounds of
  // the Node is changed.
  this.localBoundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), boundsInvalidationListener );
  this.localBoundsProperty.changeCount = boundsListenersAddedOrRemovedListener;

  // @public {TinyStaticProperty.<Bounds2>} - [mutable] Bounds just for children of this Node (and sub-trees), in the
  // "local" coordinate frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This is fired **asynchronously** (usually as part of a Display.updateDisplay()) when the childBounds of the
  // Node is changed.
  this.childBoundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), boundsInvalidationListener );
  this.childBoundsProperty.changeCount = boundsListenersAddedOrRemovedListener;

  // @public {TinyStaticProperty.<Bounds2>} - [mutable] Bounds just for this Node, in the "local" coordinate frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This event can be fired synchronously, and happens with the self-bounds of a Node is changed. This is NOT
  // like the other bounds Properties, which usually fire asynchronously
  this.selfBoundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), selfBoundsInvalidationListener );

  // @private {boolean} - Whether our localBounds have been set (with the ES5 setter/setLocalBounds()) to a custom
  // overridden value. If true, then localBounds itself will not be updated, but will instead always be the
  // overridden value.
  this._localBoundsOverridden = false;

  // @private {boolean} - [mutable] Whether invisible children will be excluded from this Node's bounds
  this._excludeInvisibleChildrenFromBounds = false;

  this._boundsDirty = true; // @private {boolean} - Whether bounds needs to be recomputed to be valid.
  this._localBoundsDirty = true; // @private {boolean} - Whether localBounds needs to be recomputed to be valid.
  this._selfBoundsDirty = true; // @private {boolean} - Whether selfBounds needs to be recomputed to be valid.
  this._childBoundsDirty = true; // @private {boolean} - Whether childBounds needs to be recomputed to be valid.

  if ( assert ) {
    // for assertions later to ensure that we are using the same Bounds2 copies as before
    this._originalBounds = this.boundsProperty._value;
    this._originalLocalBounds = this.localBoundsProperty._value;
    this._originalSelfBounds = this.selfBoundsProperty._value;
    this._originalChildBounds = this.childBoundsProperty._value;
  }

  // @public (scenery-internal) {Array.<Filter>}
  this._filters = [];

  // @public (scenery-internal) {Object} - Where rendering-specific settings are stored. They are generally modified
  // internally, so there is no ES5 setter for hints.
  this._hints = {
    // {number} - What type of renderer should be forced for this Node. Uses the internal bitmask structure declared
    //            in scenery.js and Renderer.js.
    renderer: DEFAULT_OPTIONS.renderer === null ? 0 : Renderer.fromName( DEFAULT_OPTIONS.renderer ),

    // {boolean} - Whether it is anticipated that opacity will be switched on. If so, having this set to true will
    //             make switching back-and-forth between opacity:1 and other opacities much faster.
    usesOpacity: DEFAULT_OPTIONS.usesOpacity,

    // {boolean} - Whether layers should be split before and after this Node.
    layerSplit: DEFAULT_OPTIONS.layerSplit,

    // {boolean} - Whether this Node and its subtree should handle transforms by using a CSS transform of a div.
    cssTransform: DEFAULT_OPTIONS.cssTransform,

    // {boolean} - When rendered as Canvas, whether we should use full (device) resolution on retina-like devices.
    //             TODO: ensure that this is working? 0.2 may have caused a regression.
    fullResolution: false,

    // {boolean} - Whether SVG (or other) content should be excluded from the DOM tree when invisible
    //             (instead of just being hidden)
    excludeInvisible: DEFAULT_OPTIONS.excludeInvisible,

    // {number|null} - If non-null, a multiplier to the detected pixel-to-pixel scaling of the WebGL Canvas
    webglScale: DEFAULT_OPTIONS.webglScale,

    // {boolean} - If true, Scenery will not fit any blocks that contain drawables attached to Nodes underneath this
    //             Node's subtree. This will typically prevent Scenery from triggering bounds computation for this
    //             sub-tree, and movement of this Node or its descendants will never trigger the refitting of a block.
    preventFit: DEFAULT_OPTIONS.preventFit
  };

  // @public {TinyEmitter} - This is fired only once for any single operation that may change the children of a Node.
  // For example, if a Node's children are [ a, b ] and setChildren( [ a, x, y, z ] ) is called on it, the
  // childrenChanged event will only be fired once after the entire operation of changing the children is completed.
  this.childrenChangedEmitter = new TinyEmitter();

  // @public {TinyEmitter} - For every single added child Node, emits with {Node} Node, {number} indexOfChild
  this.childInsertedEmitter = new TinyEmitter();

  // @public {TinyEmitter} - For every single removed child Node, emits with {Node} Node, {number} indexOfChild
  this.childRemovedEmitter = new TinyEmitter();

  // @public {TinyEmitter} - Provides a given range that may be affected by the reordering. Emits with
  // {number} minChangedIndex, {number} maxChangedIndex
  this.childrenReorderedEmitter = new TinyEmitter();

  // @public {TinyEmitter} - Fired synchronously when the transform (transformation matrix) of a Node is changed. Any
  // change to a Node's translation/rotation/scale/etc. will trigger this event.
  this.transformEmitter = new TinyEmitter();

  // @public {TinyEmitter} - Should be emitted when we need to check full metadata updates directly on Instances,
  // to see if we need to change drawable types, etc.
  this.instanceRefreshEmitter = new TinyEmitter();

  // @public {TinyEmitter} - Emitted to when we need to potentially recompute our renderer summary (bitmask flags, or
  // things that could affect descendants)
  this.rendererSummaryRefreshEmitter = new TinyEmitter();

  // @public {TinyEmitter} - Emitted to when we change filters (either opacity or generalized filters)
  this.filterChangeEmitter = new TinyEmitter();

  // @public {TinyEmitter} - Fired when the accessible Displays for this Node have changed (see PDOMInstance)
  this.accessibleDisplaysEmitter = new TinyEmitter();

  // @public {TinyEmitter}
  this.focusHighlightChangedEmitter = new TinyEmitter();

  // @public {TinyEmitter}
  this.changedInstanceEmitter = new TinyEmitter(); // emits with {Instance}, {boolean} added

  // compose accessibility
  this.initializeAccessibility();

  // @public (scenery-internal) {number} - A bitmask which specifies which renderers this Node (and only this Node,
  // not its subtree) supports.
  this._rendererBitmask = Renderer.bitmaskNodeDefault;

  // @public (scenery-internal) {RendererSummary} - A bitmask-like summary of what renderers and options are supported
  // by this Node and all of its descendants
  this._rendererSummary = new RendererSummary( this );

  /*
   * So we can traverse only the subtrees that require bounds validation for events firing.
   * This is a sum of the number of events requiring bounds validation on this Node, plus the number of children whose
   * count is non-zero.
   * NOTE: this means that if A has a child B, and B has a boundsEventCount of 5, it only contributes 1 to A's count.
   * This allows us to have changes localized (increasing B's count won't change A or any of A's ancestors), and
   * guarantees that we will know whether a subtree has bounds listeners. Also important: decreasing B's
   * boundsEventCount down to 0 will allow A to decrease its count by 1, without having to check its other children
   * (if we were just using a boolean value, this operation would require A to check if any OTHER children besides
   * B had bounds listeners)
   */
  this._boundsEventCount = 0; // @private {number}
  // @private {number} - This signals that we can validateBounds() on this subtree and we don't have to traverse further
  this._boundsEventSelfCount = 0;

  // Initialize sub-components
  this._picker = new Picker( this );

  // @public (scenery-internal) {boolean} - There are certain specific cases (in this case due to a11y) where we need
  // to know that a Node is getting removed from its parent BUT that process has not completed yet. It would be ideal
  // to not need this.
  this._isGettingRemovedFromParent = false;

  PhetioObject.call( this );

  if ( options ) {
    this.mutate( options );
  }
}

scenery.register( 'Node', Node );

inherit( PhetioObject, Node, {
  /**
   * This is an array of property (setter) names for Node.mutate(), which are also used when creating Nodes with
   * parameter objects.
   * @protected
   *
   * E.g. new scenery.Node( { x: 5, rotation: 20 } ) will create a Path, and apply setters in the order below
   * (node.x = 5; node.rotation = 20)
   *
   * Some special cases exist (for function names). new scenery.Node( { scale: 2 } ) will actually call
   * node.scale( 2 ).
   *
   * The order below is important! Don't change this without knowing the implications.
   *
   * NOTE: Translation-based mutators come before rotation/scale, since typically we think of their operations
   *       occurring "after" the rotation / scaling
   * NOTE: left/right/top/bottom/centerX/centerY are at the end, since they rely potentially on rotation / scaling
   *       changes of bounds that may happen beforehand
   */
  _mutatorKeys: NODE_OPTION_KEYS,

  /**
   * {Array.<String>} - List of all dirty flags that should be available on drawables created from this Node (or
   *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
   *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
   * @public (scenery-internal)
   *
   * Should be overridden by subtypes.
   */
  drawableMarkFlags: [],

  /**
   * Inserts a child Node at a specific index.
   * @public
   *
   * node.insertChild( 0, childNode ) will insert the child into the beginning of the children array (on the bottom
   * visually).
   *
   * node.insertChild( node.children.length, childNode ) is equivalent to node.addChild( childNode ), and appends it
   * to the end (top visually) of the children array. It is recommended to use node.addChild when possible.
   *
   * NOTE: overridden by Leaf for some subtypes
   *
   * @param {number} index - Index where the inserted child Node will be after this operation.
   * @param {Node} node - The new child to insert.
   * @param {boolean} [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  insertChild: function( index, node, isComposite ) {
    assert && assert( node !== null && node !== undefined, 'insertChild cannot insert a null/undefined child' );
    assert && assert( node instanceof Node,
      'addChild/insertChild requires the child to be a Node. Constructor: ' +
      ( node.constructor ? node.constructor.name : 'none' ) );
    assert && assert( !_.includes( this._children, node ), 'Parent already contains child' );
    assert && assert( node !== this, 'Cannot add self as a child' );
    assert && assert( node._parents !== null, 'Tried to insert a disposed child node?' );
    assert && assert( !node.isDisposed, 'Tried to insert a disposed Node' );

    // needs to be early to prevent re-entrant children modifications
    this._picker.onInsertChild( node );
    this.changeBoundsEventCount( node._boundsEventCount > 0 ? 1 : 0 );
    this._rendererSummary.summaryChange( RendererSummary.bitmaskAll, node._rendererSummary.bitmask );

    node._parents.push( this );
    this._children.splice( index, 0, node );

    // If this added subtree contains accessible content, we need to notify any relevant displays
    if ( !node._rendererSummary.isNotAccessible() ) {
      this.onAccessibleAddChild( node );
    }

    node.invalidateBounds();

    // like calling this.invalidateBounds(), but we already marked all ancestors with dirty child bounds
    this._boundsDirty = true;

    this.childInsertedEmitter.emit( node, index );

    !isComposite && this.childrenChangedEmitter.emit();

    if ( assertSlow ) { this._picker.audit(); }

    return this; // allow chaining
  },

  /**
   * Appends a child Node to our list of children.
   * @public
   *
   * The new child Node will be displayed in front (on top) of all of this node's other children.
   *
   * @param {Node} node
   * @param {boolean} [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  addChild: function( node, isComposite ) {
    this.insertChild( this._children.length, node, isComposite );

    return this; // allow chaining
  },

  /**
   * Removes a child Node from our list of children, see http://phetsims.github.io/scenery/doc/#node-removeChild
   * Will fail an assertion if the Node is not currently one of our children
   * @public
   *
   * @param {Node} node
   * @param {boolean} [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  removeChild: function( node, isComposite ) {
    assert && assert( node && node instanceof Node, 'Need to call node.removeChild() with a Node.' );
    assert && assert( this.hasChild( node ), 'Attempted to removeChild with a node that was not a child.' );

    const indexOfChild = _.indexOf( this._children, node );

    this.removeChildWithIndex( node, indexOfChild, isComposite );

    return this; // allow chaining
  },

  /**
   * Removes a child Node at a specific index (node.children[ index ]) from our list of children.
   * Will fail if the index is out of bounds.
   * @public
   *
   * @param {number} index
   * @param {boolean} [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  removeChildAt: function( index, isComposite ) {
    assert && assert( index >= 0 );
    assert && assert( index < this._children.length );

    const node = this._children[ index ];

    this.removeChildWithIndex( node, index, isComposite );

    return this; // allow chaining
  },

  /**
   * Internal method for removing a Node (always has the Node and index).
   * @private
   *
   * NOTE: overridden by Leaf for some subtypes
   *
   * @param {Node} node - The child node to remove from this Node (it's parent)
   * @param {number} indexOfChild - Should satisfy this.children[ indexOfChild ] === node
   * @param {boolean} [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   */
  removeChildWithIndex: function( node, indexOfChild, isComposite ) {
    assert && assert( node && node instanceof Node, 'Need to call node.removeChildWithIndex() with a Node.' );
    assert && assert( this.hasChild( node ), 'Attempted to removeChild with a node that was not a child.' );
    assert && assert( this._children[ indexOfChild ] === node, 'Incorrect index for removeChildWithIndex' );
    assert && assert( node._parents !== null, 'Tried to remove a disposed child node?' );

    const indexOfParent = _.indexOf( node._parents, this );

    node._isGettingRemovedFromParent = true;

    // If this added subtree contains accessible content, we need to notify any relevant displays
    // NOTE: Potentially removes bounds listeners here!
    if ( !node._rendererSummary.isNotAccessible() ) {
      this.onAccessibleRemoveChild( node );
    }

    // needs to be early to prevent re-entrant children modifications
    this._picker.onRemoveChild( node );
    this.changeBoundsEventCount( node._boundsEventCount > 0 ? -1 : 0 );
    this._rendererSummary.summaryChange( node._rendererSummary.bitmask, RendererSummary.bitmaskAll );

    node._parents.splice( indexOfParent, 1 );
    this._children.splice( indexOfChild, 1 );
    node._isGettingRemovedFromParent = false; // It is "complete"

    this.invalidateBounds();
    this._childBoundsDirty = true; // force recomputation of child bounds after removing a child

    this.childRemovedEmitter.emit( node, indexOfChild );

    !isComposite && this.childrenChangedEmitter.emit();

    if ( assertSlow ) { this._picker.audit(); }
  },

  /**
   * If a child is not at the given index, it is moved to the given index. This reorders the children of this Node so
   * that `this.children[ index ] === node`.
   * @public
   *
   * @param {Node} node - The child Node to move in the order
   * @param {number} index - The desired index (into the children array) of the child.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  moveChildToIndex: function( node, index ) {
    assert && assert( node && node instanceof Node, 'Need to call node.moveChildToIndex() with a Node.' );
    assert && assert( this.hasChild( node ), 'Attempted to moveChildToIndex with a node that was not a child.' );
    assert && assert( typeof index === 'number' && index % 1 === 0 && index >= 0 && index < this._children.length,
      'Invalid index: ' + index );

    const currentIndex = this.indexOfChild( node );
    if ( this._children[ index ] !== node ) {

      // Apply the actual children change
      this._children.splice( currentIndex, 1 );
      this._children.splice( index, 0, node );

      if ( !this._rendererSummary.isNotAccessible() ) {
        this.onAccessibleReorderedChildren();
      }

      this.childrenReorderedEmitter.emit( Math.min( currentIndex, index ), Math.max( currentIndex, index ) );
      this.childrenChangedEmitter.emit();
    }

    return this;
  },

  /**
   * Removes all children from this Node.
   * @public
   *
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  removeAllChildren: function() {
    this.setChildren( [] );

    return this; // allow chaining
  },

  /**
   * Sets the children of the Node to be equivalent to the passed-in array of Nodes.
   * @public
   *
   * NOTE: Overridden in LayoutBox
   *
   * @param {Array.<Node>} children
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setChildren: function( children ) {
    // The implementation is split into basically three stages:
    // 1. Remove current children that are not in the new children array.
    // 2. Reorder children that exist both before/after the change.
    // 3. Insert in new children

    const beforeOnly = []; // Will hold all nodes that will be removed.
    const afterOnly = []; // Will hold all nodes that will be "new" children (added)
    const inBoth = []; // Child nodes that "stay". Will be ordered for the "after" case.
    let i;

    // Compute what things were added, removed, or stay.
    arrayDifference( children, this._children, afterOnly, beforeOnly, inBoth );

    // Remove any nodes that are not in the new children.
    for ( i = beforeOnly.length - 1; i >= 0; i-- ) {
      this.removeChild( beforeOnly[ i ], true );
    }

    assert && assert( this._children.length === inBoth.length,
      'Removing children should not have triggered other children changes' );

    // Handle the main reordering (of nodes that "stay")
    let minChangeIndex = -1; // What is the smallest index where this._children[ index ] !== inBoth[ index ]
    let maxChangeIndex = -1; // What is the largest index where this._children[ index ] !== inBoth[ index ]
    for ( i = 0; i < inBoth.length; i++ ) {
      const desired = inBoth[ i ];
      if ( this._children[ i ] !== desired ) {
        this._children[ i ] = desired;
        if ( minChangeIndex === -1 ) {
          minChangeIndex = i;
        }
        maxChangeIndex = i;
      }
    }
    // If our minChangeIndex is still -1, then none of those nodes that "stay" were reordered. It's important to check
    // for this case, so that `node.children = node.children` is effectively a no-op performance-wise.
    const hasReorderingChange = minChangeIndex !== -1;

    // Immediate consequences/updates from reordering
    if ( hasReorderingChange ) {
      if ( !this._rendererSummary.isNotAccessible() ) {
        this.onAccessibleReorderedChildren();
      }

      this.childrenReorderedEmitter.emit( minChangeIndex, maxChangeIndex );
    }

    // Add in "new" children.
    // Scan through the "ending" children indices, adding in things that were in the "afterOnly" part. This scan is
    // done through the children array instead of the afterOnly array (as determining the index in children would
    // then be quadratic in time, which would be unacceptable here). At this point, a forward scan should be
    // sufficient to insert in-place, and should move the least amount of nodes in the array.
    if ( afterOnly.length ) {
      let afterIndex = 0;
      let after = afterOnly[ afterIndex ];
      for ( i = 0; i < children.length; i++ ) {
        if ( children[ i ] === after ) {
          this.insertChild( i, after, true );
          after = afterOnly[ ++afterIndex ];
        }
      }
    }

    // If we had any changes, send the generic "changed" event.
    if ( beforeOnly.length !== 0 || afterOnly.length !== 0 || hasReorderingChange ) {
      this.childrenChangedEmitter.emit();
    }

    // Sanity checks to make sure our resulting children array is correct.
    if ( assert ) {
      for ( let j = 0; j < this._children.length; j++ ) {
        assert( children[ j ] === this._children[ j ],
          'Incorrect child after setChildren, possibly a reentrancy issue' );
      }
    }

    // allow chaining
    return this;
  },
  set children( value ) { this.setChildren( value ); },

  /**
   * Returns a defensive copy of the array of direct children of this node, ordered by what is in front (nodes at
   * the end of the arry are in front of nodes at the start).
   * @public
   *
   * Making changes to the returned result will not affect this node's children.
   *
   * @returns {Array.<Node>}
   */
  getChildren: function() {
    // TODO: ensure we are not triggering this in Scenery code when not necessary!
    return this._children.slice( 0 ); // create a defensive copy
  },
  get children() { return this.getChildren(); },

  /**
   * Returns a count of children, without needing to make a defensive copy.
   * @public
   *
   * @returns {number}
   */
  getChildrenCount: function() {
    return this._children.length;
  },

  /**
   * Returns a defensive copy of our parents. This is an array of parent nodes that is returned in no particular
   * order (as order is not important here).
   * @public
   *
   * NOTE: Modifying the returned array will not in any way modify this node's parents.
   *
   * @returns {Array.<Node>}
   */
  getParents: function() {
    return this._parents.slice( 0 ); // create a defensive copy
  },
  get parents() { return this.getParents(); },

  /**
   * Returns a single parent if it exists, otherwise null (no parents), or an assertion failure (multiple parents).
   * @public
   *
   * @returns {Node|null}
   */
  getParent: function() {
    assert && assert( this._parents.length <= 1, 'Cannot call getParent on a node with multiple parents' );
    return this._parents.length ? this._parents[ 0 ] : null;
  },
  get parent() { return this.getParent(); },

  /**
   * Gets the child at a specific index into the children array.
   * @public
   *
   * @param {number} index
   * @returns {Node}
   */
  getChildAt: function( index ) {
    return this._children[ index ];
  },

  /**
   * Finds the index of a parent Node in the parents array.
   * @public
   *
   * @param {Node} parent - Should be a parent of this node.
   * @returns {number} - An index such that this.parents[ index ] === parent
   */
  indexOfParent: function( parent ) {
    return _.indexOf( this._parents, parent );
  },

  /**
   * Finds the index of a child Node in the children array.
   * @public
   *
   * @param {Node} child - Should be a child of this node.
   * @returns {number} - An index such that this.children[ index ] === child
   */
  indexOfChild: function( child ) {
    return _.indexOf( this._children, child );
  },

  /**
   * Moves this Node to the front (end) of all of its parents children array.
   * @public
   *
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  moveToFront: function() {
    const self = this;
    _.each( this._parents.slice(), function( parent ) {
      parent.moveChildToFront( self );
    } );

    return this; // allow chaining
  },

  /**
   * Moves one of our children to the front (end) of our children array.
   * @public
   *
   * @param {Node} child - Our child to move to the front.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  moveChildToFront: function( child ) {
    return this.moveChildToIndex( child, this._children.length - 1 );
  },

  /**
   * Move this node one index forward in each of its parents.  If the Node is already at the front, this is a no-op.
   * @returns {Node}
   * @public
   */
  moveForward: function() {
    this._parents.forEach( parent => parent.moveChildForward( this ) ); // TODO: Do we need slice like moveToFront has?
    return this; // chaining
  },

  /**
   * Moves the specified child forward by one index.  If the child is already at the front, this is a no-op.
   * @param {Node} child
   * @returns {Node}
   * @public
   */
  moveChildForward: function( child ) {
    const index = this.indexOfChild( child );
    if ( index < this.getChildrenCount() - 1 ) {
      this.moveChildToIndex( child, index + 1 );
    }
    return this; // chaining
  },

  /**
   * Move this node one index backward in each of its parents.  If the Node is already at the back, this is a no-op.
   * @returns {Node}
   * @public
   */
  moveBackward: function() {
    this._parents.forEach( parent => parent.moveChildBackward( this ) ); // TODO: Do we need slice like moveToFront has?
    return this; // chaining
  },

  /**
   * Moves the specified child forward by one index.  If the child is already at the back, this is a no-op.
   * @param {Node} child
   * @returns {Node}
   * @public
   */
  moveChildBackward: function( child ) {
    const index = this.indexOfChild( child );
    if ( index > 0 ) {
      this.moveChildToIndex( child, index - 1 );
    }
    return this; // chaining
  },

  /**
   * Moves this Node to the back (front) of all of its parents children array.
   * @public
   *
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  moveToBack: function() {
    const self = this;
    _.each( this._parents.slice(), function( parent ) {
      parent.moveChildToBack( self );
    } );

    return this; // allow chaining
  },

  /**
   * Moves one of our children to the back (front) of our children array.
   * @public
   *
   * @param {Node} child - Our child to move to the back.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  moveChildToBack: function( child ) {
    return this.moveChildToIndex( child, 0 );
  },

  /**
   * Replace a child in this node's children array with another node. If the old child had accessible focus and
   * the new child is focusable, the new child will receive focus after it is added.
   *
   * @param {Node} oldChild
   * @param {Node} newChild
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  replaceChild: function( oldChild, newChild ) {
    assert && assert( oldChild instanceof Node, 'child to replace must be a Node' );
    assert && assert( newChild instanceof Node, 'new child must be a Node' );
    assert && assert( this.hasChild( oldChild ), 'Attempted to replace a node that was not a child.' );

    // information that needs to be restored
    const index = this.indexOfChild( oldChild );
    const oldChildFocused = oldChild.focused;

    this.removeChild( oldChild, true );
    this.insertChild( index, newChild, true );

    this.childrenChangedEmitter.emit();

    if ( oldChildFocused && newChild.focusable ) {
      newChild.focus();
    }

    return this; // allow chaining
  },

  /**
   * Removes this Node from all of its parents.
   * @public
   *
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  detach: function() {
    const self = this;
    _.each( this._parents.slice( 0 ), function( parent ) {
      parent.removeChild( self );
    } );

    return this; // allow chaining
  },

  /**
   * Update our event count, usually by 1 or -1. See documentation on _boundsEventCount in constructor.
   * @private
   *
   * @param {number} n - How to increment/decrement the bounds event listener count
   */
  changeBoundsEventCount: function( n ) {
    if ( n !== 0 ) {
      const zeroBefore = this._boundsEventCount === 0;

      this._boundsEventCount += n;
      assert && assert( this._boundsEventCount >= 0, 'subtree bounds event count should be guaranteed to be >= 0' );

      const zeroAfter = this._boundsEventCount === 0;

      if ( zeroBefore !== zeroAfter ) {
        // parents will only have their count
        const parentDelta = zeroBefore ? 1 : -1;

        const len = this._parents.length;
        for ( let i = 0; i < len; i++ ) {
          this._parents[ i ].changeBoundsEventCount( parentDelta );
        }
      }
    }
  },

  /**
   * Ensures that the cached selfBounds of this Node is accurate. Returns true if any sort of dirty flag was set
   * before this was called.
   * @public
   *
   * @returns {boolean} - Was the self-bounds potentially updated?
   */
  validateSelfBounds: function() {
    // validate bounds of ourself if necessary
    if ( this._selfBoundsDirty ) {
      const oldSelfBounds = scratchBounds2.set( this.selfBoundsProperty._value );

      // Rely on an overloadable method to accomplish computing our self bounds. This should update
      // this.selfBounds itself, returning whether it was actually changed. If it didn't change, we don't want to
      // send a 'selfBounds' event.
      const didSelfBoundsChange = this.updateSelfBounds();
      this._selfBoundsDirty = false;

      if ( didSelfBoundsChange ) {
        this.selfBoundsProperty.notifyListeners( oldSelfBounds );
      }

      return true;
    }

    return false;
  },

  /**
   * Ensures that cached bounds stored on this Node (and all children) are accurate. Returns true if any sort of dirty
   * flag was set before this was called.
   * @public
   *
   * @returns {boolean} - Was something potentially updated?
   */
  validateBounds: function() {
    let i;
    const notificationThreshold = 1e-13;

    let wasDirtyBefore = this.validateSelfBounds();

    // We're going to directly mutate these instances
    const ourChildBounds = this.childBoundsProperty._value;
    const ourLocalBounds = this.localBoundsProperty._value;
    const ourSelfBounds = this.selfBoundsProperty._value;
    const ourBounds = this.boundsProperty._value;

    // validate bounds of children if necessary
    if ( this._childBoundsDirty ) {
      wasDirtyBefore = true;

      // have each child validate their own bounds
      i = this._children.length;
      while ( i-- ) {
        this._children[ i ].validateBounds();
      }

      // and recompute our childBounds
      const oldChildBounds = scratchBounds2.set( ourChildBounds ); // store old value in a temporary Bounds2
      ourChildBounds.set( Bounds2.NOTHING ); // initialize to a value that can be unioned with includeBounds()

      i = this._children.length;
      while ( i-- ) {
        const child = this._children[ i ];
        if ( !this._excludeInvisibleChildrenFromBounds || child.isVisible() ) {
          ourChildBounds.includeBounds( child.bounds );
        }
      }

      // run this before firing the event
      this._childBoundsDirty = false;

      if ( !ourChildBounds.equals( oldChildBounds ) ) {
        // notifies only on an actual change
        if ( !ourChildBounds.equalsEpsilon( oldChildBounds, notificationThreshold ) ) {
          this.childBoundsProperty.notifyListeners( oldChildBounds );
        }
      }
    }

    if ( this._localBoundsDirty && !this._localBoundsOverridden ) {
      wasDirtyBefore = true;

      this._localBoundsDirty = false; // we only need this to set local bounds as dirty

      const oldLocalBounds = scratchBounds2.set( ourLocalBounds ); // store old value in a temporary Bounds2

      // local bounds are a union between our self bounds and child bounds
      ourLocalBounds.set( ourSelfBounds ).includeBounds( ourChildBounds );

      // apply clipping to the bounds if we have a clip area (all done in the local coordinate frame)
      if ( this.hasClipArea() ) {
        ourLocalBounds.constrainBounds( this.clipArea.bounds );
      }

      if ( !ourLocalBounds.equals( oldLocalBounds ) ) {
        // sanity check, see https://github.com/phetsims/scenery/issues/1071, we're running this before the localBounds
        // listeners are notified, to support limited re-entrance.
        this._boundsDirty = true;

        if ( !ourLocalBounds.equalsEpsilon( oldLocalBounds, notificationThreshold ) ) {
          this.localBoundsProperty.notifyListeners( oldLocalBounds );
        }
      }

      // adjust our transform to match maximum bounds if necessary on a local bounds change
      if ( this._maxWidth !== null || this._maxHeight !== null ) {
        this.updateMaxDimension( ourLocalBounds );
      }
    }

    // TODO: layout here?

    if ( this._boundsDirty ) {
      wasDirtyBefore = true;

      // run this before firing the event
      this._boundsDirty = false;

      const oldBounds = scratchBounds2.set( ourBounds ); // store old value in a temporary Bounds2

      // no need to do the more expensive bounds transformation if we are still axis-aligned
      if ( this._transformBounds && !this._transform.getMatrix().isAxisAligned() ) {
        // mutates the matrix and bounds during recursion

        const matrix = scratchMatrix3.set( this.getMatrix() ); // calls below mutate this matrix
        ourBounds.set( Bounds2.NOTHING );
        // Include each painted self individually, transformed with the exact transform matrix.
        // This is expensive, as we have to do 2 matrix transforms for every descendant.
        this._includeTransformedSubtreeBounds( matrix, ourBounds ); // self and children

        if ( this.hasClipArea() ) {
          ourBounds.constrainBounds( this.clipArea.getBoundsWithTransform( matrix ) );
        }
      }
      else {
        // converts local to parent bounds. mutable methods used to minimize number of created bounds instances
        // (we create one so we don't change references to the old one)
        ourBounds.set( ourLocalBounds );
        this.transformBoundsFromLocalToParent( ourBounds );
      }

      if ( !ourBounds.equals( oldBounds ) ) {
        // if we have a bounds change, we need to invalidate our parents so they can be recomputed
        i = this._parents.length;
        while ( i-- ) {
          this._parents[ i ].invalidateBounds();
        }

        // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
        if ( !ourBounds.equalsEpsilon( oldBounds, notificationThreshold ) ) {
          this.boundsProperty.notifyListeners( oldBounds );
        }
      }
    }

    // if there were side-effects, run the validation again until we are clean
    if ( this._childBoundsDirty || this._boundsDirty ) {
      // TODO: if there are side-effects in listeners, this could overflow the stack. we should report an error
      // instead of locking up
      this.validateBounds();
    }

    if ( assert ) {
      assert( this._originalBounds === this.boundsProperty._value, 'Reference for bounds changed!' );
      assert( this._originalLocalBounds === this.localBoundsProperty._value, 'Reference for localBounds changed!' );
      assert( this._originalSelfBounds === this.selfBoundsProperty._value, 'Reference for selfBounds changed!' );
      assert( this._originalChildBounds === this.childBoundsProperty._value, 'Reference for childBounds changed!' );
    }

    // double-check that all of our bounds handling has been accurate
    if ( assertSlow ) {
      // new scope for safety
      ( () => {
        const epsilon = 0.000001;

        const childBounds = Bounds2.NOTHING.copy();
        _.each( this._children, child => {
          if ( !this._excludeInvisibleChildrenFromBounds || child.isVisible() ) {
            childBounds.includeBounds( child.boundsProperty._value );
          }
        } );

        let localBounds = this.selfBoundsProperty._value.union( childBounds );

        if ( this.hasClipArea() ) {
          localBounds = localBounds.intersection( this.clipArea.bounds );
        }

        const fullBounds = this.localToParentBounds( localBounds );

        assertSlow && assertSlow( this.childBoundsProperty._value.equalsEpsilon( childBounds, epsilon ),
          'Child bounds mismatch after validateBounds: ' +
          this.childBoundsProperty._value.toString() + ', expected: ' + childBounds.toString() );
        assertSlow && assertSlow( this._localBoundsOverridden ||
                                  this._transformBounds ||
                                  this.boundsProperty._value.equalsEpsilon( fullBounds, epsilon ) ||
                                  this.boundsProperty._value.equalsEpsilon( fullBounds, epsilon ),
          'Bounds mismatch after validateBounds: ' + this.boundsProperty._value.toString() +
          ', expected: ' + fullBounds.toString() );
      } )();
    }

    return wasDirtyBefore; // whether any dirty flags were set
  },

  /**
   * Recursion for accurate transformed bounds handling. Mutates bounds with the added bounds.
   * Mutates the matrix (parameter), but mutates it back to the starting point (within floating-point error).
   * @private
   *
   * @param {Matrix3} matrix
   * @param {Bounds2} bounds
   */
  _includeTransformedSubtreeBounds: function( matrix, bounds ) {
    if ( !this.selfBounds.isEmpty() ) {
      bounds.includeBounds( this.getTransformedSelfBounds( matrix ) );
    }

    const numChildren = this._children.length;
    for ( let i = 0; i < numChildren; i++ ) {
      const child = this._children[ i ];

      matrix.multiplyMatrix( child._transform.getMatrix() );
      child._includeTransformedSubtreeBounds( matrix, bounds );
      matrix.multiplyMatrix( child._transform.getInverse() );
    }

    return bounds;
  },

  /**
   * Traverses this subtree and validates bounds only for subtrees that have bounds listeners (trying to exclude as
   * much as possible for performance). This is done so that we can do the minimum bounds validation to prevent any
   * bounds listeners from being triggered in further validateBounds() calls without other Node changes being done.
   * This is required for Display's atomic (non-reentrant) updateDisplay(), so that we don't accidentally trigger
   * bounds listeners while computing bounds during updateDisplay().
   * @public (scenery-internal)
   *
   * NOTE: this should pass by (ignore) any overridden localBounds, to trigger listeners below.
   */
  validateWatchedBounds: function() {
    // Since a bounds listener on one of the roots could invalidate bounds on the other, we need to keep running this
    // until they are all clean. Otherwise, side-effects could occur from bounds validations
    // TODO: consider a way to prevent infinite loops here that occur due to bounds listeners triggering cycles
    while ( this.watchedBoundsScan() ) {
      // do nothing
    }
  },

  /**
   * Recursive function for validateWatchedBounds. Returned whether any validateBounds() returned true (means we have
   * to traverse again)
   * @public (scenery-internal)
   *
   * @returns {boolean} - Whether there could have been any changes.
   */
  watchedBoundsScan: function() {
    if ( this._boundsEventSelfCount !== 0 ) {
      // we are a root that should be validated. return whether we updated anything
      return this.validateBounds();
    }
    else if ( this._boundsEventCount > 0 && this._childBoundsDirty ) {
      // descendants have watched bounds, traverse!
      let changed = false;
      const numChildren = this._children.length;
      for ( let i = 0; i < numChildren; i++ ) {
        changed = this._children[ i ].watchedBoundsScan() || changed;
      }
      return changed;
    }
    else {
      // if _boundsEventCount is zero, no bounds are watched below us (don't traverse), and it wasn't changed
      return false;
    }
  },

  /**
   * Marks the bounds of this Node as invalid, so they are recomputed before being accessed again.
   * @public
   */
  invalidateBounds: function() {
    // TODO: sometimes we won't need to invalidate local bounds! it's not too much of a hassle though?
    this._boundsDirty = true;
    this._localBoundsDirty = true;

    // and set flags for all ancestors
    let i = this._parents.length;
    while ( i-- ) {
      this._parents[ i ].invalidateChildBounds();
    }
  },

  /**
   * Recursively tag all ancestors with _childBoundsDirty
   * @public (scenery-internal)
   */
  invalidateChildBounds: function() {
    // don't bother updating if we've already been tagged
    if ( !this._childBoundsDirty ) {
      this._childBoundsDirty = true;
      this._localBoundsDirty = true;
      let i = this._parents.length;
      while ( i-- ) {
        this._parents[ i ].invalidateChildBounds();
      }
    }
  },

  /**
   * Should be called to notify that our selfBounds needs to change to this new value.
   * @public
   *
   * @param {Bounds2} [newSelfBounds]
   */
  invalidateSelf: function( newSelfBounds ) {
    assert && assert( newSelfBounds === undefined || newSelfBounds instanceof Bounds2,
      'invalidateSelf\'s newSelfBounds, if provided, needs to be Bounds2' );

    const ourSelfBounds = this.selfBoundsProperty._value;

    // If no self bounds are provided, rely on the bounds validation to trigger computation (using updateSelfBounds()).
    if ( !newSelfBounds ) {
      this._selfBoundsDirty = true;
      this.invalidateBounds();
      this._picker.onSelfBoundsDirty();
    }
    // Otherwise, set the self bounds directly
    else {
      assert && assert( newSelfBounds.isEmpty() || newSelfBounds.isFinite(), 'Bounds must be empty or finite in invalidateSelf' );

      // Don't recompute the self bounds
      this._selfBoundsDirty = false;

      // if these bounds are different than current self bounds
      if ( !ourSelfBounds.equals( newSelfBounds ) ) {
        const oldSelfBounds = scratchBounds2.set( ourSelfBounds );

        // set repaint flags
        this.invalidateBounds();
        this._picker.onSelfBoundsDirty();

        // record the new bounds
        ourSelfBounds.set( newSelfBounds );

        // fire the event immediately
        this.selfBoundsProperty.notifyListeners( oldSelfBounds );
      }
    }

    if ( assertSlow ) { this._picker.audit(); }
  },

  /**
   * Meant to be overridden by Node sub-types to compute self bounds (if invalidateSelf() with no argments was called).
   * @protected
   *
   * @returns {boolean} - Whether the self bounds changed.
   */
  updateSelfBounds: function() {
    // The Node implementation (un-overridden) will never change the self bounds (always NOTHING).
    assert && assert( this.selfBoundsProperty._value.equals( Bounds2.NOTHING ) );
    return false;
  },

  /**
   * Returns whether a Node is a child of this node.
   * @public
   *
   * @param {Node} potentialChild
   * @returns {boolean} - Whether potentialChild is actually our child.
   */
  hasChild: function( potentialChild ) {
    assert && assert( potentialChild && ( potentialChild instanceof Node ), 'hasChild needs to be called with a Node' );
    const isOurChild = _.includes( this._children, potentialChild );
    assert && assert( isOurChild === _.includes( potentialChild._parents, this ), 'child-parent reference should match parent-child reference' );
    return isOurChild;
  },

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   * @public
   *
   * @returns {Shape}
   */
  getSelfShape: function() {
    const selfBounds = this.selfBounds;
    if ( selfBounds.isEmpty() ) {
      return new Shape();
    }
    else {
      return Shape.bounds( this.selfBounds );
    }
  },

  /**
   * Returns our selfBounds (the bounds for this Node's content in the local coordinates, excluding anything from our
   * children and descendants).
   * @public
   *
   * NOTE: Do NOT mutate the returned value!
   *
   * @returns {Bounds2}
   */
  getSelfBounds: function() {
    return this.selfBoundsProperty.value;
  },
  get selfBounds() { return this.getSelfBounds(); },

  /**
   * Returns a bounding box that should contain all self content in the local coordinate frame (our normal self bounds
   * aren't guaranteed this for Text, etc.)
   * @public
   *
   * Override this to provide different behavior.
   *
   * @returns {Bounds2}
   */
  getSafeSelfBounds: function() {
    return this.selfBoundsProperty.value;
  },
  get safeSelfBounds() { return this.getSafeSelfBounds(); },

  /**
   * Returns the bounding box that should contain all content of our children in our local coordinate frame. Does not
   * include our "self" bounds.
   * @public
   *
   * NOTE: Do NOT mutate the returned value!
   *
   * @returns {Bounds2}
   */
  getChildBounds: function() {
    return this.childBoundsProperty.value;
  },
  get childBounds() { return this.getChildBounds(); },

  /**
   * Returns the bounding box that should contain all content of our children AND our self in our local coordinate
   * frame.
   * @public
   *
   * NOTE: Do NOT mutate the returned value!
   *
   * @returns {Bounds2}
   */
  getLocalBounds: function() {
    return this.localBoundsProperty.value;
  },
  get localBounds() { return this.getLocalBounds(); },

  /**
   * Allows overriding the value of localBounds (and thus changing things like 'bounds' that depend on localBounds).
   * If it's set to a non-null value, that value will always be used for localBounds until this function is called
   * again. To revert to having Scenery compute the localBounds, set this to null.  The bounds should not be reduced
   * smaller than the visible bounds on the screen.
   * @public
   *
   * @param {Bounds2|null} localBounds
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setLocalBounds: function( localBounds ) {
    assert && assert( localBounds === null || localBounds instanceof Bounds2, 'localBounds override should be set to either null or a Bounds2' );
    assert && assert( localBounds === null || !isNaN( localBounds.minX ), 'minX for localBounds should not be NaN' );
    assert && assert( localBounds === null || !isNaN( localBounds.minY ), 'minY for localBounds should not be NaN' );
    assert && assert( localBounds === null || !isNaN( localBounds.maxX ), 'maxX for localBounds should not be NaN' );
    assert && assert( localBounds === null || !isNaN( localBounds.maxY ), 'maxY for localBounds should not be NaN' );

    const ourLocalBounds = this.localBoundsProperty._value;

    if ( localBounds === null ) {
      // we can just ignore this if we weren't actually overriding local bounds before
      if ( this._localBoundsOverridden ) {
        this._localBoundsOverridden = false;
        this.invalidateBounds();
      }
    }
    else {
      // just an instance check for now. consider equals() in the future depending on cost
      const changed = !localBounds.equals( ourLocalBounds ) || !this._localBoundsOverridden;

      if ( changed ) {
        ourLocalBounds.set( localBounds );
      }

      if ( !this._localBoundsOverridden ) {
        this._localBoundsOverridden = true; // NOTE: has to be done before invalidating bounds, since this disables localBounds computation
      }

      if ( changed ) {
        this.invalidateBounds();
      }
    }

    return this; // allow chaining
  },
  set localBounds( value ) { return this.setLocalBounds( value ); },

  /**
   * Meant to be overridden in sub-types that have more accurate bounds determination for when we are transformed.
   * Usually rotation is significant here, so that transformed bounds for non-rectangular shapes will be different.
   * @public
   *
   * @param {Matrix3} matrix
   * @returns {Bounds2}
   */
  getTransformedSelfBounds: function( matrix ) {
    // assume that we take up the entire rectangular bounds by default
    return this.selfBounds.transformed( matrix );
  },

  /**
   * Meant to be overridden in sub-types that have more accurate bounds determination for when we are transformed.
   * Usually rotation is significant here, so that transformed bounds for non-rectangular shapes will be different.
   * @public
   *
   * This should include the "full" bounds that guarantee everything rendered should be inside (e.g. Text, where the
   * normal bounds may not be sufficient).
   *
   * @param {Matrix3} matrix
   * @returns {Bounds2}
   */
  getTransformedSafeSelfBounds: function( matrix ) {
    return this.safeSelfBounds.transformed( matrix );
  },

  /**
   * Returns the visual "safe" bounds that are taken up by this Node and its subtree. Notably, this is essentially the
   * combined effects of the "visible" bounds (i.e. invisible nodes do not contribute to bounds), and "safe" bounds
   * (e.g. Text, where we need a larger bounds area to guarantee there is nothing outside). It also tries to "fit"
   * transformed bounds more tightly, where it will handle rotated Path bounds in an improved way.
   * @public
   *
   * NOTE: This method is not optimized, and may create garbage and not be the fastest.
   *
   * @param {Matrix3} [matrix] - If provided, will return the bounds assuming the content is transformed with the
   *                             given matrix.
   * @returns {Bounds2}
   */
  getSafeTransformedVisibleBounds: function( matrix ) {
    const localMatrix = ( matrix || Matrix3.IDENTITY ).timesMatrix( this.matrix );

    const bounds = Bounds2.NOTHING.copy();

    if ( this.visibleProperty.value ) {
      if ( !this.selfBounds.isEmpty() ) {
        bounds.includeBounds( this.getTransformedSafeSelfBounds( localMatrix ) );
      }

      if ( this._children.length ) {
        for ( let i = 0; i < this._children.length; i++ ) {
          bounds.includeBounds( this._children[ i ].getSafeTransformedVisibleBounds( localMatrix ) );
        }
      }
    }

    return bounds;
  },
  get safeTransformedVisibleBounds() { return this.getSafeTransformedVisibleBounds(); },

  /**
   * Sets the flag that determines whether we will require more accurate (and expensive) bounds computation for this
   * node's transform.
   * @public
   *
   * If set to false (default), Scenery will get the bounds of content, and then if rotated will determine the on-axis
   * bounds that completely cover the rotated bounds (potentially larger than actual content).
   * If set to true, Scenery will try to get the bounds of the actual rotated/transformed content.
   *
   * A good example of when this is necessary is if there are a bunch of nested children that each have pi/4 rotations.
   *
   * @param {boolean} transformBounds - Whether accurate transform bounds should be used.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setTransformBounds: function( transformBounds ) {
    assert && assert( typeof transformBounds === 'boolean', 'transformBounds should be boolean' );

    if ( this._transformBounds !== transformBounds ) {
      this._transformBounds = transformBounds;

      this.invalidateBounds();
    }

    return this; // allow chaining
  },
  set transformBounds( value ) { return this.setTransformBounds( value ); },

  /**
   * Returns whether accurate transformation bounds are used in bounds computation (see setTransformBounds).
   * @public
   *
   * @returns {boolean}
   */
  getTransformBounds: function() {
    return this._transformBounds;
  },
  get transformBounds() { return this.getTransformBounds(); },

  /**
   * Returns the bounding box of this Node and all of its sub-trees (in the "parent" coordinate frame).
   *
   * NOTE: Do NOT mutate the returned value!
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {Bounds2}
   */
  getBounds: function() {
    return this.boundsProperty.value;
  },
  get bounds() { return this.getBounds(); },

  /**
   * Like getLocalBounds() in the "local" coordinate frame, but includes only visible nodes.
   * @public
   *
   * @returns {Bounds2}
   */
  getVisibleLocalBounds: function() {
    // defensive copy, since we use mutable modifications below
    const bounds = this.selfBounds.copy();

    let i = this._children.length;
    while ( i-- ) {
      const child = this._children[ i ];
      if ( child.isVisible() ) {
        bounds.includeBounds( child.getVisibleBounds() );
      }
    }

    assert && assert( bounds.isFinite() || bounds.isEmpty(), 'Visible bounds should not be infinite' );
    return bounds;
  },
  get visibleLocalBounds() { return this.getVisibleLocalBounds(); },

  /**
   * Like getBounds() in the "parent" coordinate frame, but includes only visible nodes
   * @public
   *
   * @returns {Bounds2}
   */
  getVisibleBounds: function() {
    return this.getVisibleLocalBounds().transform( this.getMatrix() );
  },
  get visibleBounds() { return this.getVisibleBounds(); },

  /**
   * Tests whether the given point is "contained" in this node's subtree (optionally using mouse/touch areas), and if
   * so returns the Trail (rooted at this node) to the top-most (in stacking order) Node that contains the given
   * point.
   * @public
   *
   * NOTE: This is optimized for the current input system (rather than what gets visually displayed on the screen), so
   * pickability (Node's pickable property, visibility, and the presence of input listeners) all may affect the
   * returned value.
   *
   * For example, hit-testing a simple shape (with no pickability) will return null:
   * > new scenery.Circle( 20 ).hitTest( dot.v2( 0, 0 ) ); // null
   *
   * If the same shape is made to be pickable, it will return a trail:
   * > new scenery.Circle( 20, { pickable: true } ).hitTest( dot.v2( 0, 0 ) );
   * > // returns a Trail with the circle as the only node.
   *
   * It will return the result that is visually stacked on top, so e.g.:
   * > new scenery.Node( {
   * >   pickable: true,
   * >   children: [
   * >     new scenery.Circle( 20 ),
   * >     new scenery.Circle( 15 )
   * >   ]
   * > } ).hitTest( dot.v2( 0, 0 ) ); // returns the "top-most" circle (the one with radius:15).
   *
   * This is used by Scenery's internal input system by calling hitTest on a Display's rootNode with the
   * global-coordinate point.
   *
   * @param {Vector2} point - The point (in the parent coordinate frame) to check against this node's subtree.
   * @param {boolean} [isMouse] - Whether mouseAreas should be used.
   * @param {boolean} [isTouch] - Whether touchAreas should be used.
   * @returns {Trail|null} - Returns null if the point is not contained in the subtree.
   */
  hitTest: function( point, isMouse, isTouch ) {
    assert && assert( point instanceof Vector2 && point.isFinite(), 'The point should be a finite Vector2' );
    assert && assert( isMouse === undefined || typeof isMouse === 'boolean',
      'If isMouse is provided, it should be a boolean' );
    assert && assert( isTouch === undefined || typeof isTouch === 'boolean',
      'If isTouch is provided, it should be a boolean' );

    return this._picker.hitTest( point, !!isMouse, !!isTouch );
  },

  /**
   * Hit-tests what is under the pointer, and returns a {Trail} to that Node (or null if there is no matching node).
   * @public
   *
   * See hitTest() for more details about what will be returned.
   *
   * @param {Pointer} pointer
   * @returns {Trail|null}
   */
  trailUnderPointer: function( pointer ) {
    return this.hitTest( pointer.point, pointer instanceof Mouse, pointer instanceof Touch || pointer instanceof Pen );
  },

  /**
   * Returns whether a point (in parent coordinates) is contained in this node's sub-tree.
   * @public
   *
   * See hitTest() for more details about what will be returned.
   *
   * @param {Vector2} point
   * @returns {boolean} - Whether the point is contained.
   */
  containsPoint: function( point ) {
    return this.hitTest( point ) !== null;
  },

  /**
   * Override this for computation of whether a point is inside our self content (defaults to selfBounds check).
   * @protected
   *
   * @param {Vector2} point - Considered to be in the local coordinate frame
   * @returns {boolean}
   */
  containsPointSelf: function( point ) {
    // if self bounds are not null default to checking self bounds
    return this.selfBounds.containsPoint( point );
  },

  /**
   * Returns whether this node's selfBounds is intersected by the specified bounds.
   * @public
   *
   * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
   * @returns {boolean}
   */
  intersectsBoundsSelf: function( bounds ) {
    // if self bounds are not null, child should override this
    return this.selfBounds.intersectsBounds( bounds );
  },

  /**
   * Whether this Node itself is painted (displays something itself). Meant to be overridden.
   * @public
   *
   * @returns {boolean}
   */
  isPainted: function() {
    // Normal nodes don't render anything
    return false;
  },

  /**
   * Whether this Node's selfBounds are considered to be valid (always containing the displayed self content
   * of this node). Meant to be overridden in subtypes when this can change (e.g. Text).
   * @public
   *
   * If this value would potentially change, please trigger the event 'selfBoundsValid'.
   *
   * @returns {boolean}
   */
  areSelfBoundsValid: function() {
    return true;
  },

  /**
   * Returns whether this Node has any parents at all.
   * @public
   *
   * @returns {boolean}
   */
  hasParent: function() {
    return this._parents.length !== 0;
  },

  /**
   * Returns whether this Node has any children at all.
   * @public
   *
   * @returns {boolean}
   */
  hasChildren: function() {
    return this._children.length > 0;
  },

  /**
   * Returns whether a child should be included for layout (if this Node is a layout container).
   * @public
   *
   * @param {Node} child
   * @returns {boolean}
   */
  isChildIncludedInLayout: function( child ) {
    assert && assert( child instanceof Node );

    return child.bounds.isValid() && ( !this._excludeInvisibleChildrenFromBounds || child.visible );
  },

  /**
   * Calls the callback on nodes recursively in a depth-first manner.
   * @public
   *
   * @param {Function} callback
   */
  walkDepthFirst: function( callback ) {
    callback( this );
    const length = this._children.length;
    for ( let i = 0; i < length; i++ ) {
      this._children[ i ].walkDepthFirst( callback );
    }
  },

  /**
   * Adds an input listener.
   * @public
   *
   * See Input.js documentation for information about how event listeners are used.
   *
   * @param {Object} listener
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  addInputListener: function( listener ) {
    assert && assert( !_.includes( this._inputListeners, listener ), 'Input listener already registered on this Node' );
    assert && assert( listener !== null, 'Input listener cannot be null' );
    assert && assert( listener !== undefined, 'Input listener cannot be undefined' );

    // don't allow listeners to be added multiple times
    if ( !_.includes( this._inputListeners, listener ) ) {
      this._inputListeners.push( listener );
      this._picker.onAddInputListener();
      if ( assertSlow ) { this._picker.audit(); }
    }
    return this;
  },

  /**
   * Removes an input listener that was previously added with addInputListener.
   * @public
   *
   * @param {Object} listener
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  removeInputListener: function( listener ) {
    const index = _.indexOf( this._inputListeners, listener );

    // ensure the listener is in our list (ignore assertion for disposal, see https://github.com/phetsims/sun/issues/394)
    assert && assert( this.isDisposed || index >= 0, 'Could not find input listener to remove' );
    if ( index >= 0 ) {
      this._inputListeners.splice( index, 1 );
      this._picker.onRemoveInputListener();
      if ( assertSlow ) { this._picker.audit(); }
    }

    return this;
  },

  /**
   * Returns whether this input listener is currently listening to this node.
   * @public
   *
   * More efficient than checking node.inputListeners, as that includes a defensive copy.
   *
   * @param {Object} listener
   * @returns {boolean}
   */
  hasInputListener: function( listener ) {
    for ( let i = 0; i < this._inputListeners.length; i++ ) {
      if ( this._inputListeners[ i ] === listener ) {
        return true;
      }
    }
    return false;
  },

  /**
   * Interrupts all input listeners that are attached to this node.
   * @public
   *
   * @returns {Node} - For chaining
   */
  interruptInput: function() {
    const listenersCopy = this.inputListeners;

    for ( let i = 0; i < listenersCopy.length; i++ ) {
      const listener = listenersCopy[ i ];

      listener.interrupt && listener.interrupt();
    }

    return this;
  },

  /**
   * Interrupts all input listeners that are attached to either this node, or a descendant node.
   * @public
   *
   * @returns {Node} - For chaining
   */
  interruptSubtreeInput: function() {
    this.interruptInput();

    const children = this._children.slice();
    for ( let i = 0; i < children.length; i++ ) {
      children[ i ].interruptSubtreeInput();
    }

    return this;
  },

  /**
   * Changes the transform of this Node by adding a transform. The default "appends" the transform, so that it will
   * appear to happen to the Node before the rest of the transform would apply, but if "prepended", the rest of the
   * transform would apply first.
   * @public
   *
   * As an example, if a Node is centered at (0,0) and scaled by 2:
   * translate( 100, 0 ) would cause the center of the Node (in the parent coordinate frame) to be at (200,0).
   * translate( 100, 0, true ) would cause the center of the Node (in the parent coordinate frame) to be at (100,0).
   *
   * Allowed call signatures:
   * translate( x {number}, y {number} )
   * translate( x {number}, y {number}, prependInstead {boolean} )
   * translate( vector {Vector2} )
   * translate( vector {Vector2}, prependInstead {boolean} )
   *
   * @param {number} x - The x coordinate
   * @param {number} y - The y coordinate
   * @param {boolean} [prependInstead] - Whether the transform should be prepended (defaults to false)
   */
  translate: function( x, y, prependInstead ) {
    if ( typeof x === 'number' ) {
      // translate( x, y, prependInstead )
      assert && assert( typeof x === 'number' && isFinite( x ), 'x should be a finite number' );
      assert && assert( typeof y === 'number' && isFinite( y ), 'y should be a finite number' );
      assert && assert( prependInstead === undefined || typeof prependInstead === 'boolean', 'If provided, prependInstead should be boolean' );

      if ( !x && !y ) { return; } // bail out if both are zero
      if ( prependInstead ) {
        this.prependTranslation( x, y );
      }
      else {
        this.appendMatrix( scratchMatrix3.setToTranslation( x, y ) );
      }
    }
    else {
      // translate( vector, prependInstead )
      const vector = x;
      assert && assert( vector instanceof Vector2 && vector.isFinite(), 'translation should be a finite Vector2 if not finite numbers' );
      if ( !vector.x && !vector.y ) { return; } // bail out if both are zero
      this.translate( vector.x, vector.y, y ); // forward to full version
    }
  },

  /**
   * Scales the node's transform. The default "appends" the transform, so that it will
   * appear to happen to the Node before the rest of the transform would apply, but if "prepended", the rest of the
   * transform would apply first.
   * @public
   *
   * As an example, if a Node is translated to (100,0):
   * scale( 2 ) will leave the Node translated at (100,0), but it will be twice as big around its origin at that location.
   * scale( 2, true ) will shift the Node to (200,0).
   *
   * Allowed call signatures:
   * scale( s {number|Vector2}, [prependInstead] {boolean} )
   * scale( x {number}, y {number}, [prependInstead] {boolean} )
   *
   * @param {number} s - Scales in both the X and Y directions
   * @param {number} x - Scales in the X direction
   * @param {number} y - Scales in the Y direction
   * @param {boolean} [prependInstead] - Whether the transform should be prepended (defaults to false)
   */
  scale: function( x, y, prependInstead ) {
    if ( typeof x === 'number' ) {
      assert && assert( isFinite( x ), 'scales should be finite' );
      if ( y === undefined || typeof y === 'boolean' ) {
        // scale( scale, [prependInstead] )
        this.scale( x, x, y );
      }
      else {
        // scale( x, y, [prependInstead] )
        assert && assert( typeof y === 'number' && isFinite( y ), 'scales should be finite numbers' );
        assert && assert( prependInstead === undefined || typeof prependInstead === 'boolean', 'If provided, prependInstead should be boolean' );

        if ( x === 1 && y === 1 ) { return; } // bail out if we are scaling by 1 (identity)
        if ( prependInstead ) {
          this.prependMatrix( Matrix3.scaling( x, y ) );
        }
        else {
          this.appendMatrix( Matrix3.scaling( x, y ) );
        }
      }
    }
    else {
      // scale( vector, [prependInstead] )
      const vector = x;
      assert && assert( vector instanceof Vector2 && vector.isFinite(), 'scale should be a finite Vector2 if not a finite number' );
      this.scale( vector.x, vector.y, y ); // forward to full version
    }
  },

  /**
   * Rotates the node's transform. The default "appends" the transform, so that it will
   * appear to happen to the Node before the rest of the transform would apply, but if "prepended", the rest of the
   * transform would apply first.
   * @public
   *
   * As an example, if a Node is translated to (100,0):
   * rotate( Math.PI ) will rotate the Node around (100,0)
   * rotate( Math.PI, true ) will rotate the Node around the origin, moving it to (-100,0)
   *
   * @param {number} angle - The angle (in radians) to rotate by
   * @param {boolean} [prependInstead] - Whether the transform should be prepended (defaults to false)
   */
  rotate: function( angle, prependInstead ) {
    assert && assert( typeof angle === 'number' && isFinite( angle ), 'angle should be a finite number' );
    assert && assert( prependInstead === undefined || typeof prependInstead === 'boolean' );
    if ( angle % ( 2 * Math.PI ) === 0 ) { return; } // bail out if our angle is effectively 0
    if ( prependInstead ) {
      this.prependMatrix( Matrix3.rotation2( angle ) );
    }
    else {
      this.appendMatrix( Matrix3.rotation2( angle ) );
    }
  },

  /**
   * Rotates the node's transform around a specific point (in the parent coordinate frame) by prepending the transform.
   * @public
   *
   * TODO: determine whether this should use the appendMatrix method
   *
   * @param {Vector2} point - In the parent coordinate frame
   * @param {number} angle - In radians
   * @returns {Node} - For chaining
   */
  rotateAround: function( point, angle ) {
    assert && assert( point instanceof Vector2 && point.isFinite(), 'point should be a finite Vector2' );
    assert && assert( typeof angle === 'number' && isFinite( angle ), 'angle should be a finite number' );

    let matrix = Matrix3.translation( -point.x, -point.y );
    matrix = Matrix3.rotation2( angle ).timesMatrix( matrix );
    matrix = Matrix3.translation( point.x, point.y ).timesMatrix( matrix );
    this.prependMatrix( matrix );
    return this;
  },

  /**
   * Shifts the x coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
   * @public
   *
   * @param {number} x
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setX: function( x ) {
    assert && assert( typeof x === 'number' && isFinite( x ), 'x should be a finite number' );

    this.translate( x - this.getX(), 0, true );
    return this;
  },
  set x( value ) { this.setX( value ); },

  /**
   * Returns the x coordinate (in the parent coorindate frame) of where the node's origin is transformed to.
   * @public
   *
   * @returns {number}
   */
  getX: function() {
    return this._transform.getMatrix().m02();
  },
  get x() { return this.getX(); },

  /**
   * Shifts the y coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
   * @public
   *
   * @param {number} y
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setY: function( y ) {
    assert && assert( typeof y === 'number' && isFinite( y ), 'y should be a finite number' );

    this.translate( 0, y - this.getY(), true );
    return this;
  },
  set y( value ) { this.setY( value ); },

  /**
   * Returns the y coordinate (in the parent coorindate frame) of where the node's origin is transformed to.
   * @public
   *
   * @returns {number}
   */
  getY: function() {
    return this._transform.getMatrix().m12();
  },
  get y() { return this.getY(); },

  /**
   * Typically without rotations or negative parameters, this sets the scale for each axis. In its more general form,
   * it modifies the node's transform so that:
   * - Transforming (1,0) with our transform will result in a vector with magnitude abs( x-scale-magnitude )
   * - Transforming (0,1) with our transform will result in a vector with magnitude abs( y-scale-magnitude )
   * - If parameters are negative, it will flip orientation in that direct.
   * @public
   *
   * Allowed call signatures:
   * setScaleMagnitude( s )
   * setScaleMagnitude( sx, sy )
   * setScaleMagnitude( vector )
   *
   * @param {number} s - Scale for both axes
   * @param {number} sx - Scale for the X axis
   * @param {number} sy - Scale for the Y axis
   * @param {Vector2} vector - Scale for the x/y axes in the vector's components.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setScaleMagnitude: function( a, b ) {
    const currentScale = this.getScaleVector();

    if ( typeof a === 'number' ) {
      if ( b === undefined ) {
        // to map setScaleMagnitude( scale ) => setScaleMagnitude( scale, scale )
        b = a;
      }
      assert && assert( typeof a === 'number' && isFinite( a ), 'setScaleMagnitude parameters should be finite numbers' );
      assert && assert( typeof b === 'number' && isFinite( b ), 'setScaleMagnitude parameters should be finite numbers' );
      // setScaleMagnitude( x, y )
      this.appendMatrix( Matrix3.scaling( a / currentScale.x, b / currentScale.y ) );
    }
    else {
      // setScaleMagnitude( vector ), where we set the x-scale to vector.x and y-scale to vector.y
      assert && assert( a instanceof Vector2 && a.isFinite(), 'first parameter should be a finite Vector2' );

      this.appendMatrix( Matrix3.scaling( a.x / currentScale.x, a.y / currentScale.y ) );
    }
    return this;
  },

  /**
   * Returns a vector with an entry for each axis, e.g. (5,2) for an affine matrix with rows ((5,0,0),(0,2,0),(0,0,1)).
   * @public
   *
   * It is equivalent to:
   * ( T(1,0).magnitude(), T(0,1).magnitude() ) where T() transforms points with our transform.
   *
   * @returns {Vector2}
   */
  getScaleVector: function() {
    return this._transform.getMatrix().getScaleVector();
  },

  /**
   * Rotates this node's transform so that a unit (1,0) vector would be rotated by this node's transform by the
   * specified amount.
   * @public
   *
   * @param {number} rotation - In radians
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setRotation: function( rotation ) {
    assert && assert( typeof rotation === 'number' && isFinite( rotation ),
      'rotation should be a finite number' );

    this.appendMatrix( scratchMatrix3.setToRotationZ( rotation - this.getRotation() ) );
    return this;
  },
  set rotation( value ) { this.setRotation( value ); },

  /**
   * Returns the rotation (in radians) that would be applied to a unit (1,0) vector when transformed with this Node's
   * transform.
   * @public
   *
   * @returns {number}
   */
  getRotation: function() {
    return this._transform.getMatrix().getRotation();
  },
  get rotation() { return this.getRotation(); },

  /**
   * Modifies the translation of this Node's transform so that the node's local-coordinate origin will be transformed
   * to the passed-in x/y.
   * @public
   *
   * Allowed call signatures:
   * setTranslation( x, y )
   * setTranslation( vector )
   *
   * @param {number} x - X translation
   * @param {number} y - Y translation
   * @param {Vector2} vector - Vector with x/y translation in components
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setTranslation: function( a, b ) {
    const m = this._transform.getMatrix();
    const tx = m.m02();
    const ty = m.m12();

    let dx;
    let dy;

    if ( typeof a === 'number' ) {
      assert && assert( typeof a === 'number' && isFinite( a ), 'Parameters to setTranslation should be finite numbers' );
      assert && assert( typeof b === 'number' && isFinite( b ), 'Parameters to setTranslation should be finite numbers' );
      dx = a - tx;
      dy = b - ty;
    }
    else {
      assert && assert( a instanceof Vector2 && a.isFinite(), 'Should be a finite Vector2' );
      dx = a.x - tx;
      dy = a.y - ty;
    }

    this.translate( dx, dy, true );

    return this;
  },
  set translation( value ) { this.setTranslation( value ); },

  /**
   * Returns a vector of where this Node's local-coordinate origin will be transformed by it's own transform.
   * @public
   *
   * @returns {Vector2}
   */
  getTranslation: function() {
    const matrix = this._transform.getMatrix();
    return new Vector2( matrix.m02(), matrix.m12() );
  },
  get translation() { return this.getTranslation(); },

  /**
   * Appends a transformation matrix to this Node's transform. Appending means this transform is conceptually applied
   * first before the rest of the Node's current transform (i.e. applied in the local coordinate frame).
   * @public
   *
   * @param {Matrix3} matrix
   */
  appendMatrix: function( matrix ) {
    assert && assert( matrix instanceof Matrix3 && matrix.isFinite(), 'matrix should be a finite Matrix3' );
    assert && assert( matrix.getDeterminant() !== 0, 'matrix should not map plane to a line or point' );
    this._transform.append( matrix );
  },

  /**
   * Prepends a transformation matrix to this Node's transform. Prepending means this transform is conceptually applied
   * after the rest of the Node's current transform (i.e. applied in the parent coordinate frame).
   * @public
   *
   * @param {Matrix3} matrix
   */
  prependMatrix: function( matrix ) {
    assert && assert( matrix instanceof Matrix3 && matrix.isFinite(), 'matrix should be a finite Matrix3' );
    assert && assert( matrix.getDeterminant() !== 0, 'matrix should not map plane to a line or point' );
    this._transform.prepend( matrix );
  },

  /**
   * Prepends an (x,y) translation to our Node's transform in an efficient manner without allocating a matrix.
   * see https://github.com/phetsims/scenery/issues/119
   * @public
   *
   * @param {number} x
   * @param {number} y
   */
  prependTranslation: function( x, y ) {
    assert && assert( typeof x === 'number' && isFinite( x ), 'x should be a finite number' );
    assert && assert( typeof y === 'number' && isFinite( y ), 'y should be a finite number' );

    if ( !x && !y ) { return; } // bail out if both are zero

    this._transform.prependTranslation( x, y );
  },

  /**
   * Changes this Node's transform to match the passed-in transformation matrix.
   * @public
   *
   * @param {Matrix3} matrix
   */
  setMatrix: function( matrix ) {
    assert && assert( matrix instanceof Matrix3 && matrix.isFinite(), 'matrix should be a finite Matrix3' );
    assert && assert( matrix.getDeterminant() !== 0, 'matrix should not map plane to a line or point' );

    this._transform.setMatrix( matrix );
  },
  set matrix( value ) { this.setMatrix( value ); },

  /**
   * Returns a Matrix3 representing our Node's transform.
   * @public
   *
   * NOTE: Do not mutate the returned matrix.
   *
   * @returns {Matrix3}
   */
  getMatrix: function() {
    return this._transform.getMatrix();
  },
  get matrix() { return this.getMatrix(); },

  /**
   * Returns a reference to our Node's transform
   * @public
   *
   * @returns {Transform3}
   */
  getTransform: function() {
    // for now, return an actual copy. we can consider listening to changes in the future
    return this._transform;
  },
  get transform() { return this.getTransform(); },

  /**
   * Resets our Node's transform to an identity transform (i.e. no transform is applied).
   * @public
   */
  resetTransform: function() {
    this.setMatrix( Matrix3.IDENTITY );
  },

  /**
   * Callback function that should be called when our transform is changed.
   * @private
   */
  onTransformChange: function() {
    // TODO: why is local bounds invalidation needed here?
    this.invalidateBounds();

    this._picker.onTransformChange();
    if ( assertSlow ) { this._picker.audit(); }

    this.transformEmitter.emit();
  },

  /**
   * Called when our summary bitmask changes
   * @public (scenery-internal)
   *
   * @param {number} oldBitmask
   * @param {number} newBitmask
   */
  onSummaryChange: function( oldBitmask, newBitmask ) {
    // Defined in ParallelDOM.js
    this._accessibleDisplaysInfo.onSummaryChange( oldBitmask, newBitmask );
  },

  /**
   * Updates our node's scale and applied scale factor if we need to change our scale to fit within the maximum
   * dimensions (maxWidth and maxHeight). See documentation in constructor for detailed behavior.
   * @private
   *
   * @param {Bounds2} localBounds
   */
  updateMaxDimension: function( localBounds ) {
    const currentScale = this._appliedScaleFactor;
    let idealScale = 1;

    if ( this._maxWidth !== null ) {
      const width = localBounds.width;
      if ( width > this._maxWidth ) {
        idealScale = Math.min( idealScale, this._maxWidth / width );
      }
    }

    if ( this._maxHeight !== null ) {
      const height = localBounds.height;
      if ( height > this._maxHeight ) {
        idealScale = Math.min( idealScale, this._maxHeight / height );
      }
    }

    const scaleAdjustment = idealScale / currentScale;
    if ( scaleAdjustment !== 1 ) {
      this.scale( scaleAdjustment );

      this._appliedScaleFactor = idealScale;
    }
  },

  /**
   * Increments/decrements bounds "listener" count based on the values of maxWidth/maxHeight before and after.
   * null is like no listener, non-null is like having a listener, so we increment for null => non-null, and
   * decrement for non-null => null.
   * @private
   *
   * @param {null | number} beforeMaxLength
   * @param {null | number} afterMaxLength
   */
  onMaxDimensionChange: function( beforeMaxLength, afterMaxLength ) {
    if ( beforeMaxLength === null && afterMaxLength !== null ) {
      this.changeBoundsEventCount( 1 );
      this._boundsEventSelfCount++;
    }
    else if ( beforeMaxLength !== null && afterMaxLength === null ) {
      this.changeBoundsEventCount( -1 );
      this._boundsEventSelfCount--;
    }
  },

  /**
   * Sets the maximum width of the Node (see constructor for documentation on how maximum dimensions work).
   * @public
   *
   * @param {number|null} maxWidth
   */
  setMaxWidth: function( maxWidth ) {
    assert && assert( maxWidth === null || ( typeof maxWidth === 'number' && maxWidth > 0 ),
      'maxWidth should be null (no constraint) or a positive number' );

    if ( this._maxWidth !== maxWidth ) {
      // update synthetic bounds listener count (to ensure our bounds are validated at the start of updateDisplay)
      this.onMaxDimensionChange( this._maxWidth, maxWidth );

      this._maxWidth = maxWidth;

      this.updateMaxDimension( this.localBoundsProperty.value );
    }
  },
  set maxWidth( value ) { this.setMaxWidth( value ); },

  /**
   * Returns the maximum width (if any) of the Node.
   * @public
   *
   * @returns {number|null}
   */
  getMaxWidth: function() {
    return this._maxWidth;
  },
  get maxWidth() { return this.getMaxWidth(); },

  /**
   * Sets the maximum height of the Node (see constructor for documentation on how maximum dimensions work).
   * @public
   *
   * @param {number|null} maxHeight
   */
  setMaxHeight: function( maxHeight ) {
    assert && assert( maxHeight === null || ( typeof maxHeight === 'number' && maxHeight > 0 ),
      'maxHeight should be null (no constraint) or a positive number' );

    if ( this._maxHeight !== maxHeight ) {
      // update synthetic bounds listener count (to ensure our bounds are validated at the start of updateDisplay)
      this.onMaxDimensionChange( this._maxHeight, maxHeight );

      this._maxHeight = maxHeight;

      this.updateMaxDimension( this.localBoundsProperty.value );
    }
  },
  set maxHeight( value ) { this.setMaxHeight( value ); },

  /**
   * Returns the maximum height (if any) of the Node.
   * @public
   *
   * @returns {number|null}
   */
  getMaxHeight: function() {
    return this._maxHeight;
  },
  get maxHeight() { return this.getMaxHeight(); },

  /**
   * Shifts this Node horizontally so that its left bound (in the parent coordinate frame) is equal to the passed-in
   * 'left' X value.
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param {number} left - After this operation, node.left should approximately equal this value.
   * @returns {Node} - For chaining
   */
  setLeft: function( left ) {
    assert && assert( typeof left === 'number' );

    const currentLeft = this.getLeft();
    if ( isFinite( currentLeft ) ) {
      this.translate( left - currentLeft, 0, true );
    }

    return this; // allow chaining
  },
  set left( value ) { this.setLeft( value ); },

  /**
   * Returns the X value of the left side of the bounding box of this Node (in the parent coordinate frame).
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getLeft: function() {
    return this.getBounds().minX;
  },
  get left() { return this.getLeft(); },

  /**
   * Shifts this Node horizontally so that its right bound (in the parent coordinate frame) is equal to the passed-in
   * 'right' X value.
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param {number} right - After this operation, node.right should approximately equal this value.
   * @returns {Node} - For chaining
   */
  setRight: function( right ) {
    assert && assert( typeof right === 'number' );

    const currentRight = this.getRight();
    if ( isFinite( currentRight ) ) {
      this.translate( right - currentRight, 0, true );
    }
    return this; // allow chaining
  },
  set right( value ) { this.setRight( value ); },

  /**
   * Returns the X value of the right side of the bounding box of this Node (in the parent coordinate frame).
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getRight: function() {
    return this.getBounds().maxX;
  },
  get right() { return this.getRight(); },

  /**
   * Shifts this Node horizontally so that its horizontal center (in the parent coordinate frame) is equal to the
   * passed-in center X value.
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param {number} x - After this operation, node.centerX should approximately equal this value.
   * @returns {Node} - For chaining
   */
  setCenterX: function( x ) {
    assert && assert( typeof x === 'number' );

    const currentCenterX = this.getCenterX();
    if ( isFinite( currentCenterX ) ) {
      this.translate( x - currentCenterX, 0, true );
    }

    return this; // allow chaining
  },
  set centerX( value ) { this.setCenterX( value ); },

  /**
   * Returns the X value of this node's horizontal center (in the parent coordinate frame)
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getCenterX: function() {
    return this.getBounds().getCenterX();
  },
  get centerX() { return this.getCenterX(); },

  /**
   * Shifts this Node vertically so that its vertical center (in the parent coordinate frame) is equal to the
   * passed-in center Y value.
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param {number} y - After this operation, node.centerY should approximately equal this value.
   * @returns {Node} - For chaining
   */
  setCenterY: function( y ) {
    assert && assert( typeof y === 'number' );

    const currentCenterY = this.getCenterY();
    if ( isFinite( currentCenterY ) ) {
      this.translate( 0, y - currentCenterY, true );
    }

    return this; // allow chaining
  },
  set centerY( value ) { this.setCenterY( value ); },

  /**
   * Returns the Y value of this node's vertical center (in the parent coordinate frame)
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getCenterY: function() {
    return this.getBounds().getCenterY();
  },
  get centerY() { return this.getCenterY(); },

  /**
   * Shifts this Node vertically so that its top (in the parent coordinate frame) is equal to the passed-in Y value.
   * @public
   *
   * NOTE: top is the lowest Y value in our bounds.
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param {number} top - After this operation, node.top should approximately equal this value.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setTop: function( top ) {
    assert && assert( typeof top === 'number' );

    const currentTop = this.getTop();
    if ( isFinite( currentTop ) ) {
      this.translate( 0, top - currentTop, true );
    }

    return this; // allow chaining
  },
  set top( value ) { this.setTop( value ); },

  /**
   * Returns the lowest Y value of this node's bounding box (in the parent coordinate frame).
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getTop: function() {
    return this.getBounds().minY;
  },
  get top() { return this.getTop(); },

  /**
   * Shifts this Node vertically so that its bottom (in the parent coordinate frame) is equal to the passed-in Y value.
   * @public
   *
   * NOTE: bottom is the highest Y value in our bounds.
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param {number} bottom - After this operation, node.bottom should approximately equal this value.
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setBottom: function( bottom ) {
    assert && assert( typeof bottom === 'number' );

    const currentBottom = this.getBottom();
    if ( isFinite( currentBottom ) ) {
      this.translate( 0, bottom - currentBottom, true );
    }

    return this; // allow chaining
  },
  set bottom( value ) { this.setBottom( value ); },

  /**
   * Returns the highest Y value of this node's bounding box (in the parent coordinate frame).
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getBottom: function() {
    return this.getBounds().maxY;
  },
  get bottom() { return this.getBottom(); },

  /*
   * Convenience locations
   *
   * Upper is in terms of the visual layout in Scenery and other programs, so the minY is the "upper", and minY is the "lower"
   *
   *             left (x)     centerX        right
   *          ---------------------------------------
   * top  (y) | leftTop     centerTop     rightTop
   * centerY  | leftCenter  center        rightCenter
   * bottom   | leftBottom  centerBottom  rightBottom
   *
   * NOTE: This requires computation of this node's subtree bounds, which may incur some performance loss.
   */

  /**
   * Sets the position of the upper-left corner of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} leftTop
   * @returns {Node} - For chaining
   */
  setLeftTop: function( leftTop ) {
    assert && assert( leftTop instanceof Vector2 && leftTop.isFinite(), 'leftTop should be a finite Vector2' );

    const currentLeftTop = this.getLeftTop();
    if ( currentLeftTop.isFinite() ) {
      this.translate( leftTop.minus( currentLeftTop ), true );
    }

    return this;
  },
  set leftTop( value ) { this.setLeftTop( value ); },

  /**
   * Returns the upper-left corner of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getLeftTop: function() {
    return this.getBounds().getLeftTop();
  },
  get leftTop() { return this.getLeftTop(); },

  /**
   * Sets the position of the center-top location of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} centerTop
   * @returns {Node} - For chaining
   */
  setCenterTop: function( centerTop ) {
    assert && assert( centerTop instanceof Vector2 && centerTop.isFinite(), 'centerTop should be a finite Vector2' );

    const currentCenterTop = this.getCenterTop();
    if ( currentCenterTop.isFinite() ) {
      this.translate( centerTop.minus( currentCenterTop ), true );
    }

    return this;
  },
  set centerTop( value ) { this.setCenterTop( value ); },

  /**
   * Returns the center-top location of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getCenterTop: function() {
    return this.getBounds().getCenterTop();
  },
  get centerTop() { return this.getCenterTop(); },

  /**
   * Sets the position of the upper-right corner of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} rightTop
   * @returns {Node} - For chaining
   */
  setRightTop: function( rightTop ) {
    assert && assert( rightTop instanceof Vector2 && rightTop.isFinite(), 'rightTop should be a finite Vector2' );

    const currentRightTop = this.getRightTop();
    if ( currentRightTop.isFinite() ) {
      this.translate( rightTop.minus( currentRightTop ), true );
    }

    return this;
  },
  set rightTop( value ) { this.setRightTop( value ); },

  /**
   * Returns the upper-right corner of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getRightTop: function() {
    return this.getBounds().getRightTop();
  },
  get rightTop() { return this.getRightTop(); },

  /**
   * Sets the position of the center-left of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} leftCenter
   * @returns {Node} - For chaining
   */
  setLeftCenter: function( leftCenter ) {
    assert && assert( leftCenter instanceof Vector2 && leftCenter.isFinite(), 'leftCenter should be a finite Vector2' );

    const currentLeftCenter = this.getLeftCenter();
    if ( currentLeftCenter.isFinite() ) {
      this.translate( leftCenter.minus( currentLeftCenter ), true );
    }

    return this;
  },
  set leftCenter( value ) { this.setLeftCenter( value ); },

  /**
   * Returns the center-left corner of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getLeftCenter: function() {
    return this.getBounds().getLeftCenter();
  },
  get leftCenter() { return this.getLeftCenter(); },

  /**
   * Sets the center of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} center
   * @returns {Node} - For chaining
   */
  setCenter: function( center ) {
    assert && assert( center instanceof Vector2 && center.isFinite(), 'center should be a finite Vector2' );

    const currentCenter = this.getCenter();
    if ( currentCenter.isFinite() ) {
      this.translate( center.minus( currentCenter ), true );
    }

    return this;
  },
  set center( value ) { this.setCenter( value ); },

  /**
   * Returns the center of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getCenter: function() {
    return this.getBounds().getCenter();
  },
  get center() { return this.getCenter(); },

  /**
   * Sets the position of the center-right of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} rightCenter
   * @returns {Node} - For chaining
   */
  setRightCenter: function( rightCenter ) {
    assert && assert( rightCenter instanceof Vector2 && rightCenter.isFinite(), 'rightCenter should be a finite Vector2' );

    const currentRightCenter = this.getRightCenter();
    if ( currentRightCenter.isFinite() ) {
      this.translate( rightCenter.minus( currentRightCenter ), true );
    }

    return this;
  },
  set rightCenter( value ) { this.setRightCenter( value ); },

  /**
   * Returns the center-right of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getRightCenter: function() {
    return this.getBounds().getRightCenter();
  },
  get rightCenter() { return this.getRightCenter(); },

  /**
   * Sets the position of the lower-left corner of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} leftBottom
   * @returns {Node} - For chaining
   */
  setLeftBottom: function( leftBottom ) {
    assert && assert( leftBottom instanceof Vector2 && leftBottom.isFinite(), 'leftBottom should be a finite Vector2' );

    const currentLeftBottom = this.getLeftBottom();
    if ( currentLeftBottom.isFinite() ) {
      this.translate( leftBottom.minus( currentLeftBottom ), true );
    }

    return this;
  },
  set leftBottom( value ) { this.setLeftBottom( value ); },

  /**
   * Returns the lower-left corner of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getLeftBottom: function() {
    return this.getBounds().getLeftBottom();
  },
  get leftBottom() { return this.getLeftBottom(); },

  /**
   * Sets the position of the center-bottom of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} centerBottom
   * @returns {Node} - For chaining
   */
  setCenterBottom: function( centerBottom ) {
    assert && assert( centerBottom instanceof Vector2 && centerBottom.isFinite(), 'centerBottom should be a finite Vector2' );

    const currentCenterBottom = this.getCenterBottom();
    if ( currentCenterBottom.isFinite() ) {
      this.translate( centerBottom.minus( currentCenterBottom ), true );
    }

    return this;
  },
  set centerBottom( value ) { this.setCenterBottom( value ); },

  /**
   * Returns the center-bottom of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getCenterBottom: function() {
    return this.getBounds().getCenterBottom();
  },
  get centerBottom() { return this.getCenterBottom(); },

  /**
   * Sets the position of the lower-right corner of this node's bounds to the specified point.
   * @public
   *
   * @param {Vector2} rightBottom
   * @returns {Node} - For chaining
   */
  setRightBottom: function( rightBottom ) {
    assert && assert( rightBottom instanceof Vector2 && rightBottom.isFinite(), 'rightBottom should be a finite Vector2' );

    const currentRightBottom = this.getRightBottom();
    if ( currentRightBottom.isFinite() ) {
      this.translate( rightBottom.minus( currentRightBottom ), true );
    }

    return this;
  },
  set rightBottom( value ) { this.setRightBottom( value ); },

  /**
   * Returns the lower-right corner of this node's bounds.
   * @public
   *
   * @returns {Vector2}
   */
  getRightBottom: function() {
    return this.getBounds().getRightBottom();
  },
  get rightBottom() { return this.getRightBottom(); },

  /**
   * Returns the width of this node's bounding box (in the parent coordinate frame).
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getWidth: function() {
    return this.getBounds().getWidth();
  },
  get width() { return this.getWidth(); },

  /**
   * Returns the height of this node's bounding box (in the parent coordinate frame).
   * @public
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {number}
   */
  getHeight: function() {
    return this.getBounds().getHeight();
  },
  get height() { return this.getHeight(); },

  /**
   * Returns the unique integral ID for this node.
   * @public
   *
   * @returns {number}
   */
  getId: function() {
    return this._id;
  },
  get id() { return this.getId(); },

  /**
   * Called when our visibility Property changes values.
   * @private
   *
   * @param {boolean} visible
   */
  onVisiblePropertyChange: function( visible ) {
    // changing visibility can affect pickability pruning, which affects mouse/touch bounds
    this._picker.onVisibilityChange();

    if ( assertSlow ) { this._picker.audit(); }

    // Defined in ParallelDOM.js
    this._accessibleDisplaysInfo.onVisibilityChange( visible );

    for ( let i = 0; i < this._parents.length; i++ ) {
      const parent = this._parents[ i ];
      if ( parent._excludeInvisibleChildrenFromBounds ) {
        parent.invalidateChildBounds();
      }
    }
  },

  /**
   * Sets what Property our visibleProperty is backed by, so that changes to this provided Property will change this
   * Node's visibility, and vice versa. This does not change this._visibleProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   *
   * Note that all instrumented Nodes create their own instrumented visibleProperty (if one is not passed in as an option).
   * Once a Node's visibleProperty has been registered with PhET-iO, it cannot be "swapped out" for another.
   *
   * @public
   *
   * @param {TinyProperty.<boolean>|Property.<boolean>|null} newTarget
   * @returns {Node} for chaining
   */
  setVisibleProperty( newTarget ) {
    return this._visibleProperty.setTargetProperty( this, VISIBLE_PROPERTY_TANDEM_NAME, newTarget );
  },
  set visibleProperty( property ) { this.setVisibleProperty( property ); },

  /**
   * Get this Node's visibleProperty. Note! This is not the reciprocal of setVisibleProperty. Node.prototype._visibleProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the visibleProperty provided by
   * setVisibleProperty(), but the underlying reference does not change. This means the following:
   *     * const myNode = new Node();
   * const visibleProperty = new Property( false );
   * myNode.setVisibleProperty( visibleProperty )
   * => myNode.getVisibleProperty() !== visibleProperty (!!!!!!)
   *
   * Please use this with caution. See setVisibleProperty() for more information.
   *
   * @returns {TinyForwardingProperty}
   */
  getVisibleProperty: function() {
    return this._visibleProperty;
  },
  get visibleProperty() { return this.getVisibleProperty(); },

  /**
   * Sets whether this Node is visible.
   * @public
   *
   * @param {boolean} visible
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  setVisible: function( visible ) {
    assert && assert( typeof visible === 'boolean', 'Node visibility should be a boolean value' );

    this.visibleProperty.set( visible );

    return this;
  },
  set visible( value ) { this.setVisible( value ); },

  /**
   * Returns whether this Node is visible.
   * @public
   *
   * @returns {boolean}
   */
  isVisible: function() {
    return this.visibleProperty.value;
  },
  get visible() { return this.isVisible(); },

  /**
   * Swap the visibility of this node with another node. The Node that is made visible will receive keyboard focus
   * if it is focusable and the previously visible Node had focus.
   * @public
   *
   * @param {Node} otherNode
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  swapVisibility: function( otherNode ) {
    assert && assert( this.visible !== otherNode.visible );

    const visibleNode = this.visible ? this : otherNode;
    const invisibleNode = this.visible ? otherNode : this;

    // if the visible node has focus we will restore focus on the invisible Node once it is visible
    const visibleNodeFocused = visibleNode.focused;

    visibleNode.visible = false;
    invisibleNode.visible = true;

    if ( visibleNodeFocused && invisibleNode.focusable ) {
      invisibleNode.focus();
    }

    return this; // allow chaining
  },

  /**
   * Sets the opacity of this Node (and its sub-tree), where 0 is fully transparent, and 1 is fully opaque.
   * @public
   *
   * NOTE: opacity is clamped to be between 0 and 1.
   *
   * @param {number} opacity
   */
  setOpacity: function( opacity ) {
    assert && assert( typeof opacity === 'number' && isFinite( opacity ), 'opacity should be a finite number' );

    this.opacityProperty.value = clamp( opacity, 0, 1 );
  },
  set opacity( value ) { this.setOpacity( value ); },

  /**
   * Returns the opacity of this node.
   * @public
   *
   * @returns {number}
   */
  getOpacity: function() {
    return this.opacityProperty.value;
  },
  get opacity() { return this.getOpacity(); },

  /**
   * Called when our opacity Property changes values.
   * @private
   *
   * @param {boolean} opacity
   * @param {boolean} oldOpacity
   */
  onOpacityPropertyChange: function( opacity, oldOpacity ) {
    this.filterChangeEmitter.emit();
  },

  /**
   * Sets the non-opacity filters for this Node.
   * @public
   *
   * @param {Array.<Filter>} filters
   */
  setFilters: function( filters ) {
    assert && assert( Array.isArray( filters ), 'filters should be an array' );
    assert && assert( _.every( filters, filter => filter instanceof Filter ), 'filters should consist of Filter objects only' );

    this._filters.length = 0;
    this._filters.push( ...filters );

    this.invalidateHint();
    this.filterChangeEmitter.emit();
  },
  set filters( value ) { this.setFilters( value ); },

  /**
   * Returns the non-opacity filters for this Node.
   * @public
   *
   * @returns {Array.<Filter>}
   */
  getFilters: function() {
    return this._filters.slice();
  },
  get filters() { return this.getFilters(); },

  /**
   * Sets what Property our pickableProperty is backed by, so that changes to this provided Property will change this
   * Node's pickability, and vice versa. This does not change this._pickableProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   *
   * Instrumented Nodes do not by default create their own instrumented pickableProperty, even though Node.visibleProperty does.
   *
   * @public
   *
   * @param {TinyProperty.<boolean>|Property.<boolean>|null} newTarget
   * @returns {Node} for chaining
   */
  setPickableProperty( newTarget ) {

    return this._pickableProperty.setTargetProperty( this, PICKABLE_PROPERTY_TANDEM_NAME, newTarget );
  },
  set pickableProperty( property ) { this.setPickableProperty( property ); },

  /**
   * Handles linking and checking child PhET-iO Properties such as visibleProperty and pickableProperty.
   * @public
   *
   * @param {string} tandemName - the name for the child tandem
   * @param {Property.<boolean>|undefined|null} oldProperty - same typedef as TinyForwardingProperty.targetProperty
   * @param {Property.<boolean>|undefined|null} newProperty - same typedef as TinyForwardingProperty.targetProperty
   */
  updateLinkedElementForProperty( tandemName, oldProperty, newProperty ) {
    assert && assert( oldProperty !== newProperty, 'should not be called on same values' );

    // Only update linked elements if this Node is instrumented for PhET-iO
    if ( this.isPhetioInstrumented() ) {

      oldProperty && oldProperty.isPhetioInstrumented() && this.removeLinkedElements( oldProperty );

      const tandem = this.tandem.createTandem( tandemName );
      if ( newProperty && newProperty.isPhetioInstrumented() && tandem !== newProperty.tandem ) {
        this.addLinkedElement( newProperty, { tandem: tandem } );
      }
    }
  },

  /**
   * Get this Node's pickableProperty. Note! This is not the reciprocal of setPickableProperty. Node.prototype._pickableProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the pickableProperty provided by
   * setPickableProperty(), but the underlying reference does not change. This means the following:
   * const myNode = new Node();
   * const pickableProperty = new Property( false );
   * myNode.setPickableProperty( pickableProperty )
   * => myNode.getPickableProperty() !== pickableProperty (!!!!!!)
   *
   * Please use this with caution. See setPickableProperty() for more information.
   *
   * @returns {TinyForwardingProperty}
   */
  getPickableProperty: function() {
    return this._pickableProperty;
  },
  get pickableProperty() { return this.getPickableProperty(); },

  /**
   * Use this to automatically create a forwarded, PhET-iO instrumented pickableProperty internal to Node. This is different
   * from visible because pickable by default doesn't not create this forwarded Property.
   * @public
   * @param {boolean} pickablePropertyPhetioInstrumented
   * @returns {Node} - for chaining
   */
  setPickablePropertyPhetioInstrumented: function( pickablePropertyPhetioInstrumented ) {
    return this._pickableProperty.setTargetPropertyInstrumented( pickablePropertyPhetioInstrumented, this );
  },
  set pickablePropertyPhetioInstrumented( pickablePropertyPhetioInstrumented ) { this.setPickablePropertyPhetioInstrumented( pickablePropertyPhetioInstrumented );},

  /**
   * Sets whether this Node (and its subtree) will allow hit-testing (and thus user interaction), controlling what
   * Trail is returned from node.trailUnderPoint().
   * @public
   *
   * Pickable can take one of three values:
   * - null: (default) pass-through behavior. Hit-testing will prune this subtree if there are no
   *         ancestors/descendants with either pickable: true set or with any input listeners.
   * - false: Hit-testing is pruned, nothing under a pickable: false will respond to events or be picked.
   * - true: Hit-testing will not be pruned in this subtree, except for pickable: false cases.
   *
   * Hit testing is accomplished mainly with node.trailUnderPointer() and node.trailUnderPoint(), following the
   * above rules. Nodes that are not pickable (pruned) will not have input events targeted to them.
   *
   * The following rules (applied in the given order) determine whether a Node (really, a Trail) will receive input events:
   * 1. If the node or one of its ancestors has pickable: false OR is invisible, the Node *will not* receive events
   *    or hit testing.
   * 2. If the Node or one of its ancestors or descendants is pickable: true OR has an input listener attached, it
   *    *will* receive events or hit testing.
   * 3. Otherwise, it *will not* receive events or hit testing.
   *
   * This is useful for semi-transparent overlays or other visual elements that should be displayed but should not
   * prevent objects below from being manipulated by user input, and the default null value is used to increase
   * performance by ignoring areas that don't need user input.
   *
   * NOTE: If you want something to be picked "mouse is over it", but block input events even if there are listeners,
   *       then pickable:false is not appropriate, and inputEnabled:false is preferred.
   *
   * For a visual example of how pickability interacts with input listeners and visibility, see the notes at the
   * bottom of http://phetsims.github.io/scenery/doc/implementation-notes, or scenery/assets/pickability.svg.
   *
   * @param {boolean|null} pickable
   * @returns {Node} - for chaining
   */
  setPickable: function( pickable ) {
    assert && assert( pickable === null || typeof pickable === 'boolean' );
    this._pickableProperty.set( pickable );

    return this;
  },
  set pickable( value ) { this.setPickable( value ); },

  /**
   * Returns the pickability of this node.
   * @public
   *
   * @returns {boolean|null}
   */
  isPickable: function() {
    return this._pickableProperty.value;
  },
  get pickable() { return this.isPickable(); },

  /**
   * Called when our visibility Property changes values.
   * @private
   *
   * @param {boolean} pickable
   * @param {boolean} oldPickable
   */
  onPickablePropertyChange: function( pickable, oldPickable ) {
    this._picker.onPickableChange( oldPickable, pickable );
    if ( assertSlow ) { this._picker.audit(); }
    // TODO: invalidate the cursor somehow? #150
  },

  /**
   * Sets whether input is enabled for this Node and its subtree. If false, input event listeners will not be fired
   * on this Node or its descendants in the picked Trail. This does NOT effect picking (what Trail/nodes are under
   * a pointer), but only effects what listeners are fired.
   * @public
   *
   * Additionally, this will affect cursor behavior. If inputEnabled=false, descendants of this Node will not be
   * checked when determining what cursor will be shown. Instead, if a pointer (e.g. mouse) is over a descendant,
   * this Node's cursor will be checked first, then ancestors will be checked as normal.
   *
   * @param {boolean} inputEnabled
   */
  setInputEnabled: function( inputEnabled ) {
    assert && assert( typeof inputEnabled === 'boolean' );

    this.inputEnabledProperty.value = inputEnabled;
  },
  set inputEnabled( value ) { this.setInputEnabled( value ); },

  /**
   * Returns whether input is enabled for this Node and its subtree. See setInputEnabled for more documentation.
   * @public
   *
   * @returns {boolean}
   */
  isInputEnabled: function() {
    return this.inputEnabledProperty.value;
  },
  get inputEnabled() { return this.isInputEnabled(); },

  /**
   * Sets all of the input listeners attached to this Node.
   * @public
   *
   * This is equivalent to removing all current input listeners with removeInputListener() and adding all new
   * listeners (in order) with addInputListener().
   *
   * @param {Array.<Object>} inputlisteners - The input listeners to add.
   * @returns {Node} - For chaining
   */
  setInputListeners: function( inputListeners ) {
    assert && assert( Array.isArray( inputListeners ) );

    // Remove all old input listeners
    while ( this._inputListeners.length ) {
      this.removeInputListener( this._inputListeners[ 0 ] );
    }

    // Add in all new input listeners
    for ( let i = 0; i < inputListeners.length; i++ ) {
      this.addInputListener( inputListeners[ i ] );
    }

    return this;
  },
  set inputListeners( value ) { this.setInputListeners( value ); },

  /**
   * Returns a copy of all of our input listeners.
   * @public
   *
   * @returns {Array.<Object>}
   */
  getInputListeners: function() {
    return this._inputListeners.slice( 0 ); // defensive copy
  },
  get inputListeners() { return this.getInputListeners(); },

  /**
   * Sets the CSS cursor string that should be used when the mouse is over this node. null is the default, and
   * indicates that ancestor nodes (or the browser default) should be used.
   * @public
   *
   * @param {string|null} cursor - A CSS cursor string, like 'pointer', or 'none'
   */
  setCursor: function( cursor ) {
    assert && assert( typeof cursor === 'string' || cursor === null );

    // TODO: consider a mapping of types to set reasonable defaults
    /*
     auto default none inherit help pointer progress wait crosshair text vertical-text alias copy move no-drop not-allowed
     e-resize n-resize w-resize s-resize nw-resize ne-resize se-resize sw-resize ew-resize ns-resize nesw-resize nwse-resize
     context-menu cell col-resize row-resize all-scroll url( ... ) --> does it support data URLs?
     */

    // allow the 'auto' cursor type to let the ancestors or scene pick the cursor type
    this._cursor = cursor === 'auto' ? null : cursor;
  },
  set cursor( value ) { this.setCursor( value ); },

  /**
   * Returns the CSS cursor string for this node, or null if there is no cursor specified.
   * @public
   *
   * @returns {string|null}
   */
  getCursor: function() {
    return this._cursor;
  },
  get cursor() { return this.getCursor(); },

  /**
   * Sets the hit-tested mouse area for this Node (see constructor for more advanced documentation). Use null for the
   * default behavior.
   * @public
   *
   * @param {Bounds2|Shape|null} area
   */
  setMouseArea: function( area ) {
    assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'mouseArea needs to be a kite.Shape, dot.Bounds2, or null' );

    if ( this._mouseArea !== area ) {
      this._mouseArea = area; // TODO: could change what is under the mouse, invalidate!

      this._picker.onMouseAreaChange();
      if ( assertSlow ) { this._picker.audit(); }
    }
  },
  set mouseArea( value ) { this.setMouseArea( value ); },

  /**
   * Returns the hit-tested mouse area for this node.
   * @public
   *
   * @returns {Bounds2|Shape|null}
   */
  getMouseArea: function() {
    return this._mouseArea;
  },
  get mouseArea() { return this.getMouseArea(); },

  /**
   * Sets the hit-tested touch area for this Node (see constructor for more advanced documentation). Use null for the
   * default behavior.
   * @public
   *
   * @param {Bounds2|Shape|null} area
   */
  setTouchArea: function( area ) {
    assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'touchArea needs to be a kite.Shape, dot.Bounds2, or null' );

    if ( this._touchArea !== area ) {
      this._touchArea = area; // TODO: could change what is under the touch, invalidate!

      this._picker.onTouchAreaChange();
      if ( assertSlow ) { this._picker.audit(); }
    }
  },
  set touchArea( value ) { this.setTouchArea( value ); },

  /**
   * Returns the hit-tested touch area for this node.
   * @public
   *
   * @returns {Bounds2|Shape|null}
   */
  getTouchArea: function() {
    return this._touchArea;
  },
  get touchArea() { return this.getTouchArea(); },

  /**
   * Sets a clipped shape where only content in our local coordinate frame that is inside the clip area will be shown
   * (anything outside is fully transparent).
   * @public
   *
   * @param {Shape|null} shape
   */
  setClipArea: function( shape ) {
    assert && assert( shape === null || shape instanceof Shape, 'clipArea needs to be a kite.Shape, or null' );

    if ( this.clipArea !== shape ) {
      this.clipAreaProperty.value = shape;

      this.invalidateBounds();
      this._picker.onClipAreaChange();

      if ( assertSlow ) { this._picker.audit(); }
    }
  },
  set clipArea( value ) { this.setClipArea( value ); },

  /**
   * Returns the clipped area for this node.
   * @public
   *
   * @returns {Shape|null}
   */
  getClipArea: function() {
    return this.clipAreaProperty.value;
  },
  get clipArea() { return this.getClipArea(); },

  /**
   * Returns whether this Node has a clip area.
   * @public
   *
   * @returns {boolean}
   */
  hasClipArea: function() {
    return this.clipArea !== null;
  },

  /**
   * Sets what self renderers (and other bitmask flags) are supported by this node.
   * @protected
   *
   * @param {number} bitmask
   */
  setRendererBitmask: function( bitmask ) {
    assert && assert( typeof bitmask === 'number' && isFinite( bitmask ) );

    if ( bitmask !== this._rendererBitmask ) {
      this._rendererBitmask = bitmask;

      this._rendererSummary.selfChange();

      this.instanceRefreshEmitter.emit();
    }
  },

  /**
   * Meant to be overridden, so that it can be called to ensure that the renderer bitmask will be up-to-date.
   * @protected
   */
  invalidateSupportedRenderers: function() {

  },

  /*---------------------------------------------------------------------------*
   * Hints
   *----------------------------------------------------------------------------*/

  /**
   * When ANY hint changes, we refresh everything currently (for safety, this may be possible to make more specific
   * in the future, but hint changes are not particularly common performance bottleneck).
   * @private
   */
  invalidateHint: function() {
    this.rendererSummaryRefreshEmitter.emit();
    this.instanceRefreshEmitter.emit();
  },

  /**
   * Sets a preferred renderer for this Node and its sub-tree. Scenery will attempt to use this renderer under here
   * unless it isn't supported, OR another preferred renderer is set as a closer ancestor. Acceptable values are:
   * - null (default, no preference)
   * - 'canvas'
   * - 'svg'
   * - 'dom'
   * - 'webgl'
   * @public
   *
   * @param {string|null} renderer
   */
  setRenderer: function( renderer ) {
    assert && assert( renderer === null || renderer === 'canvas' || renderer === 'svg' || renderer === 'dom' || renderer === 'webgl',
      'Renderer input should be null, or one of: "canvas", "svg", "dom" or "webgl".' );

    let newRenderer = 0;
    if ( renderer === 'canvas' ) {
      newRenderer = Renderer.bitmaskCanvas;
    }
    else if ( renderer === 'svg' ) {
      newRenderer = Renderer.bitmaskSVG;
    }
    else if ( renderer === 'dom' ) {
      newRenderer = Renderer.bitmaskDOM;
    }
    else if ( renderer === 'webgl' ) {
      newRenderer = Renderer.bitmaskWebGL;
    }
    assert && assert( ( renderer === null ) === ( newRenderer === 0 ),
      'We should only end up with no actual renderer if renderer is null' );

    if ( this._hints.renderer !== newRenderer ) {
      this._hints.renderer = newRenderer;

      this.invalidateHint();
    }
  },
  set renderer( value ) { this.setRenderer( value ); },

  /**
   * Returns the preferred renderer (if any) of this node, as a string.
   * @public
   *
   * @returns {string|null}
   */
  getRenderer: function() {
    if ( this._hints.renderer === 0 ) {
      return null;
    }
    else if ( this._hints.renderer === Renderer.bitmaskCanvas ) {
      return 'canvas';
    }
    else if ( this._hints.renderer === Renderer.bitmaskSVG ) {
      return 'svg';
    }
    else if ( this._hints.renderer === Renderer.bitmaskDOM ) {
      return 'dom';
    }
    else if ( this._hints.renderer === Renderer.bitmaskWebGL ) {
      return 'webgl';
    }
    assert && assert( false, 'Seems to be an invalid renderer?' );
    return this._hints.renderer;
  },
  get renderer() { return this.getRenderer(); },

  /**
   * Sets whether or not Scenery will try to put this Node (and its descendants) into a separate SVG/Canvas/WebGL/etc.
   * layer, different from other siblings or other nodes. Can be used for performance purposes.
   * @public
   *
   * @param {boolean} split
   */
  setLayerSplit: function( split ) {
    assert && assert( typeof split === 'boolean' );

    if ( split !== this._hints.layerSplit ) {
      this._hints.layerSplit = split;

      this.invalidateHint();
    }
  },
  set layerSplit( value ) { this.setLayerSplit( value ); },

  /**
   * Returns whether the layerSplit performance flag is set.
   * @public
   *
   * @returns {boolean}
   */
  isLayerSplit: function() {
    return this._hints.layerSplit;
  },
  get layerSplit() { return this.isLayerSplit(); },

  /**
   * Sets whether or not Scenery will take into account that this Node plans to use opacity. Can have performance
   * gains if there need to be multiple layers for this node's descendants.
   * @public
   *
   * @param {boolean} usesOpacity
   */
  setUsesOpacity: function( usesOpacity ) {
    assert && assert( typeof usesOpacity === 'boolean' );

    if ( usesOpacity !== this._hints.usesOpacity ) {
      this._hints.usesOpacity = usesOpacity;

      this.invalidateHint();
    }
  },
  set usesOpacity( value ) { this.setUsesOpacity( value ); },

  /**
   * Returns whether the usesOpacity performance flag is set.
   * @public
   *
   * @returns {boolean}
   */
  getUsesOpacity: function() {
    return this._hints.usesOpacity;
  },
  get usesOpacity() { return this.getUsesOpacity(); },

  /**
   * Sets a flag for whether whether the contents of this Node and its children should be displayed in a separate
   * DOM element that is transformed with CSS transforms. It can have potential speedups, since the browser may not
   * have to rerasterize contents when it is animated.
   * @public
   *
   * @param {boolean} cssTransform
   */
  setCSSTransform: function( cssTransform ) {
    assert && assert( typeof cssTransform === 'boolean' );

    if ( cssTransform !== this._hints.cssTransform ) {
      this._hints.cssTransform = cssTransform;

      this.invalidateHint();
    }
  },
  set cssTransform( value ) { this.setCSSTransform( value ); },

  /**
   * Returns wehther the cssTransform performance flag is set.
   * @public
   *
   * @returns {boolean}
   */
  isCSSTransformed: function() {
    return this._hints.cssTransform;
  },
  get cssTransform() { return this._hints.cssTransform; },

  /**
   * Sets a performance flag for whether layers/DOM elements should be excluded (or included) when things are
   * invisible. The default is false, and invisible content is in the DOM, but hidden.
   * @public
   *
   * @param {boolean} excludeInvisible
   */
  setExcludeInvisible: function( excludeInvisible ) {
    assert && assert( typeof excludeInvisible === 'boolean' );

    if ( excludeInvisible !== this._hints.excludeInvisible ) {
      this._hints.excludeInvisible = excludeInvisible;

      this.invalidateHint();
    }
  },
  set excludeInvisible( value ) { this.setExcludeInvisible( value ); },

  /**
   * Returns whether the excludeInvisible performance flag is set.
   * @public
   *
   * @returns {boolean}
   */
  isExcludeInvisible: function() {
    return this._hints.excludeInvisible;
  },
  get excludeInvisible() { return this.isExcludeInvisible(); },

  /**
   * If this is set to true, child nodes that are invisible will NOT contribute to the bounds of this node.
   * @public
   *
   * The default is for child nodes bounds' to be included in this node's bounds, but that would in general be a
   * problem for layout containers or other situations, see https://github.com/phetsims/joist/issues/608.
   *
   * @param {boolean} excludeInvisibleChildrenFromBounds
   */
  setExcludeInvisibleChildrenFromBounds: function( excludeInvisibleChildrenFromBounds ) {
    assert && assert( typeof excludeInvisibleChildrenFromBounds === 'boolean' );

    if ( excludeInvisibleChildrenFromBounds !== this._excludeInvisibleChildrenFromBounds ) {
      this._excludeInvisibleChildrenFromBounds = excludeInvisibleChildrenFromBounds;

      this.invalidateBounds();
    }
  },
  set excludeInvisibleChildrenFromBounds( value ) { this.setExcludeInvisibleChildrenFromBounds( value ); },

  /**
   * Returns whether the excludeInvisibleChildrenFromBounds flag is set, see
   * setExcludeInvisibleChildrenFromBounds() for documentation.
   * @public
   *
   * @returns {boolean}
   */
  isExcludeInvisibleChildrenFromBounds: function() {
    return this._excludeInvisibleChildrenFromBounds;
  },
  get excludeInvisibleChildrenFromBounds() { return this.isExcludeInvisibleChildrenFromBounds(); },

  setPreventFit: function( preventFit ) {
    assert && assert( typeof preventFit === 'boolean' );

    if ( preventFit !== this._hints.preventFit ) {
      this._hints.preventFit = preventFit;

      this.invalidateHint();
    }
  },
  set preventFit( value ) { this.setPreventFit( value ); },

  /**
   * Returns whether the preventFit performance flag is set.
   * @public
   *
   * @returns {boolean}
   */
  isPreventFit: function() {
    return this._hints.preventFit;
  },
  get preventFit() { return this.isPreventFit(); },

  /**
   * Sets whether there is a custom WebGL scale applied to the Canvas, and if so what scale.
   * @public
   *
   * @param {number|null} webglScale
   */
  setWebGLScale: function( webglScale ) {
    assert && assert( webglScale === null || ( typeof webglScale === 'number' && isFinite( webglScale ) ) );

    if ( webglScale !== this._hints.webglScale ) {
      this._hints.webglScale = webglScale;

      this.invalidateHint();
    }
  },
  set webglScale( value ) { this.setWebGLScale( value ); },

  /**
   * Returns the value of the webglScale performance flag.
   * @public
   *
   * @returns {number|null}
   */
  getWebGLScale: function() {
    return this._hints.webglScale;
  },
  get webglScale() { return this.getWebGLScale(); },

  /*---------------------------------------------------------------------------*
   * Trail operations
   *----------------------------------------------------------------------------*/

  /**
   * Returns the one Trail that starts from a node with no parents (or if the predicate is present, a Node that
   * satisfies it), and ends at this node. If more than one Trail would satisfy these conditions, an assertion is
   * thrown (please use getTrails() for those cases).
   * @public
   *
   * @param {function( node ) : boolean} [predicate] - If supplied, we will only return trails rooted at a Node that
   *                                                   satisfies predicate( node ) == true
   * @returns {Trail}
   */
  getUniqueTrail: function( predicate ) {

    // Without a predicate, we'll be able to bail out the instant we hit a Node with 2+ parents, and it makes the
    // logic easier.
    if ( !predicate ) {
      const trail = new Trail();
      let node = this; // eslint-disable-line consistent-this

      while ( node ) {
        assert && assert( node._parents.length <= 1,
          'getUniqueTrail found a Node with ' + node._parents.length + ' parents.' );

        trail.addAncestor( node );
        node = node._parents[ 0 ]; // should be undefined if there aren't any parents
      }

      return trail;
    }
    // With a predicate, we need to explore multiple parents (since the predicate may filter out all but one)
    else {
      const trails = this.getTrails( predicate );

      assert && assert( trails.length === 1,
        'getUniqueTrail found ' + trails.length + ' matching trails for the predicate' );

      return trails[ 0 ];
    }
  },

  /**
   * Returns a Trail rooted at rootNode and ends at this node. Throws an assertion if the number of trails that match
   * this condition isn't exactly 1.
   * @public
   *
   * @param {Node} rootNode
   * @returns {Trail}
   */
  getUniqueTrailTo: function( rootNode ) {
    return this.getUniqueTrail( function( node ) {
      return rootNode === node;
    } );
  },

  /**
   * Returns an array of all Trails that start from nodes with no parent (or if a predicate is present, those that
   * satisfy the predicate), and ends at this node.
   * @public
   *
   * @param {function( node ) : boolean} [predicate] - If supplied, we will only return Trails rooted at nodes that
   *                                                   satisfy predicate( node ) == true.
   * @returns {Array.<Trail>}
   */
  getTrails: function( predicate ) {
    predicate = predicate || Node.defaultTrailPredicate;

    const trails = [];
    const trail = new Trail( this );
    Trail.appendAncestorTrailsWithPredicate( trails, trail, predicate );

    return trails;
  },

  /**
   * Returns an array of all Trails rooted at rootNode and end at this node.
   * @public

   * @param {Node} rootNode
   * @returns {Array.<Trail>}
   */
  getTrailsTo: function( rootNode ) {
    return this.getTrails( function( node ) {
      return node === rootNode;
    } );
  },

  /**
   * Returns an array of all Trails rooted at this Node and end with nodes with no children (or if a predicate is
   * present, those that satisfy the predicate).
   * @public
   *
   * @param {function( node ) : boolean} [predicate] - If supplied, we will only return Trails ending at nodes that
   *                                                   satisfy predicate( node ) == true.
   * @returns {Array.<Trail>}
   */
  getLeafTrails: function( predicate ) {
    predicate = predicate || Node.defaultLeafTrailPredicate;

    const trails = [];
    const trail = new Trail( this );
    Trail.appendDescendantTrailsWithPredicate( trails, trail, predicate );

    return trails;
  },

  /**
   * Returns an array of all Trails rooted at this Node and end with leafNode.
   * @public
   *
   * @param {Node} leafNode
   * @returns {Array.<Trail>}
   */
  getLeafTrailsTo: function( leafNode ) {
    return this.getLeafTrails( function( node ) {
      return node === leafNode;
    } );
  },

  /**
   * Returns a Trail rooted at this node and ending at a Node that has no children (or if a predicate is provided, a
   * Node that satisfies the predicate). If more than one trail matches this description, an assertion will be fired.
   * @public
   *
   * @param {function( node ) : boolean} [predicate] - If supplied, we will return a Trail that ends with a Node that
   *                                                   satisfies predicate( node ) == true
   * @returns {Trail}
   */
  getUniqueLeafTrail: function( predicate ) {
    const trails = this.getLeafTrails( predicate );

    assert && assert( trails.length === 1,
      'getUniqueLeafTrail found ' + trails.length + ' matching trails for the predicate' );

    return trails[ 0 ];
  },

  /**
   * Returns a Trail rooted at this Node and ending at leafNode. If more than one trail matches this description,
   * an assertion will be fired.
   * @public
   *
   * @param {Node} leafNode
   * @returns {Trail}
   */
  getUniqueLeafTrailTo: function( leafNode ) {
    return this.getUniqueLeafTrail( function( node ) {
      return node === leafNode;
    } );
  },

  /**
   * Returns all nodes in the connected component, returned in an arbitrary order, including nodes that are ancestors
   * of this node.
   * @public
   *
   * @returns {Array.<Node>}
   */
  getConnectedNodes: function() {
    const result = [];
    let fresh = this._children.concat( this._parents ).concat( this );
    while ( fresh.length ) {
      const node = fresh.pop();
      if ( !_.includes( result, node ) ) {
        result.push( node );
        fresh = fresh.concat( node._children, node._parents );
      }
    }
    return result;
  },

  /**
   * Returns all nodes in the subtree with this Node as its root, returned in an arbitrary order. Like
   * getConnectedNodes, but doesn't include parents.
   * @public
   *
   * @returns {Array.<Node>}
   */
  getSubtreeNodes: function() {
    const result = [];
    let fresh = this._children.concat( this );
    while ( fresh.length ) {
      const node = fresh.pop();
      if ( !_.includes( result, node ) ) {
        result.push( node );
        fresh = fresh.concat( node._children );
      }
    }
    return result;
  },

  /**
   * Returns all nodes that are connected to this node, sorted in topological order.
   * @public
   *
   * @returns {Array.<Node>}
   */
  getTopologicallySortedNodes: function() {
    // see http://en.wikipedia.org/wiki/Topological_sorting
    const edges = {};
    const s = [];
    const l = [];
    let n;
    _.each( this.getConnectedNodes(), function( node ) {
      edges[ node.id ] = {};
      _.each( node._children, function( m ) {
        edges[ node.id ][ m.id ] = true;
      } );
      if ( !node.parents.length ) {
        s.push( node );
      }
    } );

    function handleChild( m ) {
      delete edges[ n.id ][ m.id ];
      if ( _.every( edges, function( children ) { return !children[ m.id ]; } ) ) {
        // there are no more edges to m
        s.push( m );
      }
    }

    while ( s.length ) {
      n = s.pop();
      l.push( n );

      _.each( n._children, handleChild );
    }

    // ensure that there are no edges left, since then it would contain a circular reference
    assert && assert( _.every( edges, function( children ) {
      return _.every( children, function( final ) { return false; } );
    } ), 'circular reference check' );

    return l;
  },

  /**
   * Returns whether this.addChild( child ) will not cause circular references.
   * @public
   *
   * @param {Node} child
   * @returns {boolean}
   */
  canAddChild: function( child ) {
    if ( this === child || _.includes( this._children, child ) ) {
      return false;
    }

    // see http://en.wikipedia.org/wiki/Topological_sorting
    // TODO: remove duplication with above handling?
    const edges = {};
    const s = [];
    const l = [];
    let n;
    _.each( this.getConnectedNodes().concat( child.getConnectedNodes() ), function( node ) {
      edges[ node.id ] = {};
      _.each( node._children, function( m ) {
        edges[ node.id ][ m.id ] = true;
      } );
      if ( !node.parents.length && node !== child ) {
        s.push( node );
      }
    } );
    edges[ this.id ][ child.id ] = true; // add in our 'new' edge
    function handleChild( m ) {
      delete edges[ n.id ][ m.id ];
      if ( _.every( edges, function( children ) { return !children[ m.id ]; } ) ) {
        // there are no more edges to m
        s.push( m );
      }
    }

    while ( s.length ) {
      n = s.pop();
      l.push( n );

      _.each( n._children, handleChild );

      // handle our new edge
      if ( n === this ) {
        handleChild( child );
      }
    }

    // ensure that there are no edges left, since then it would contain a circular reference
    return _.every( edges, function( children ) {
      return _.every( children, function( final ) { return false; } );
    } );
  },

  /**
   * To be overridden in paintable Node types. Should hook into the drawable's prototype (presumably).
   * @protected
   *
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param {CanvasContextWrapper} wrapper
   * @param {Matrix3} matrix - The transformation matrix already applied to the context.
   */
  canvasPaintSelf: function( wrapper, matrix ) {

  },

  /**
   * Renders this Node only (its self) into the Canvas wrapper, in its local coordinate frame.
   * @public
   *
   * @param {CanvasContextWrapper} wrapper
   * @param {Matrix3} matrix - The current transformation matrix associated with the wrapper
   */
  renderToCanvasSelf: function( wrapper, matrix ) {
    if ( this.isPainted() && ( this._rendererBitmask & Renderer.bitmaskCanvas ) ) {
      this.canvasPaintSelf( wrapper, matrix );
    }
  },

  /**
   * Renders this Node and its descendants into the Canvas wrapper.
   * @public
   *
   * @param {CanvasContextWrapper} wrapper
   * @param {Matrix3} [matrix] - Optional transform to be applied
   */
  renderToCanvasSubtree: function( wrapper, matrix ) {
    matrix = matrix || Matrix3.identity();

    wrapper.resetStyles();

    this.renderToCanvasSelf( wrapper, matrix );
    for ( let i = 0; i < this._children.length; i++ ) {
      const child = this._children[ i ];

      if ( child.isVisible() ) {
        const requiresScratchCanvas = child.opacity !== 1 || child.clipArea;

        wrapper.context.save();
        matrix.multiplyMatrix( child._transform.getMatrix() );
        matrix.canvasSetTransform( wrapper.context );
        if ( requiresScratchCanvas ) {
          const canvas = document.createElement( 'canvas' );
          canvas.width = wrapper.canvas.width;
          canvas.height = wrapper.canvas.height;
          const context = canvas.getContext( '2d' );
          const childWrapper = new CanvasContextWrapper( canvas, context );

          matrix.canvasSetTransform( context );

          child.renderToCanvasSubtree( childWrapper, matrix );

          wrapper.context.save();
          if ( child.clipArea ) {
            wrapper.context.beginPath();
            child.clipArea.writeToContext( wrapper.context );
            wrapper.context.clip();
          }
          wrapper.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
          wrapper.context.globalAlpha = child.opacity;
          wrapper.context.drawImage( canvas, 0, 0 );
          wrapper.context.restore();
        }
        else {
          child.renderToCanvasSubtree( wrapper, matrix );
        }
        matrix.multiplyMatrix( child._transform.getInverse() );
        wrapper.context.restore();
      }
    }
  },

  /**
   * @deprecated
   * Render this Node to the Canvas (clearing it first)
   * @public
   *
   * @param {HTMLCanvasElement} canvas
   * @param {CanvasRenderingContext2D} context
   * @param {Function} callback - Called with no arguments
   * @param {string} [backgroundColor]
   */
  // @public (API compatibility for now): Render this Node to the Canvas (clearing it first)
  renderToCanvas: function( canvas, context, callback, backgroundColor ) {

    assert && deprecationWarning( 'Node.renderToCanvas() is deprecated, please use Node.rasterized() instead' );

    // should basically reset everything (and clear the Canvas)
    canvas.width = canvas.width; // eslint-disable-line no-self-assign

    if ( backgroundColor ) {
      context.fillStyle = backgroundColor;
      context.fillRect( 0, 0, canvas.width, canvas.height );
    }

    const wrapper = new CanvasContextWrapper( canvas, context );

    this.renderToCanvasSubtree( wrapper, Matrix3.identity() );

    callback && callback(); // this was originally asynchronous, so we had a callback
  },

  /**
   * Renders this Node to an HTMLCanvasElement. If toCanvas( callback ) is used, the canvas will contain the node's
   * entire bounds (if no x/y/width/height is provided)
   * @public
   *
   * @param {Function} callback - callback( canvas, x, y, width, height ) is called, where x,y are computed if not specified.
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   */
  toCanvas: function( callback, x, y, width, height ) {
    assert && assert( typeof callback === 'function' );
    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    const padding = 2; // padding used if x and y are not set

    // for now, we add an unpleasant hack around Text and safe bounds in general. We don't want to add another Bounds2 object per Node for now.
    const bounds = this.getBounds().union( this.localToParentBounds( this.getSafeSelfBounds() ) );
    assert && assert( !bounds.isEmpty() ||
                      ( x !== undefined && y !== undefined && width !== undefined && height !== undefined ),
      'Should not call toCanvas on a Node with empty bounds, unless all dimensions are provided' );

    x = x !== undefined ? x : Math.ceil( padding - bounds.minX );
    y = y !== undefined ? y : Math.ceil( padding - bounds.minY );
    width = width !== undefined ? width : Math.ceil( bounds.getWidth() + 2 * padding );
    height = height !== undefined ? height : Math.ceil( bounds.getHeight() + 2 * padding );

    const canvas = document.createElement( 'canvas' );
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext( '2d' );

    // shift our rendering over by the desired amount
    context.translate( x, y );

    // for API compatibility, we apply our own transform here
    this._transform.getMatrix().canvasAppendTransform( context );

    const wrapper = new CanvasContextWrapper( canvas, context );

    this.renderToCanvasSubtree( wrapper, Matrix3.translation( x, y ).timesMatrix( this._transform.getMatrix() ) );

    callback( canvas, x, y, width, height ); // we used to be asynchronous
  },

  /**
   * Renders this Node to a Canvas, then calls the callback with the data URI from it.
   * @public
   *
   * @param {Function} callback - callback( dataURI {string}, x, y, width, height ) is called, where x,y are computed if not specified.
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   */
  toDataURL: function( callback, x, y, width, height ) {
    assert && assert( typeof callback === 'function' );
    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    this.toCanvas( function( canvas, x, y, width, height ) {
      // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
      callback( canvas.toDataURL(), x, y, width, height );
    }, x, y, width, height );
  },

  /**
   * Calls the callback with an HTMLImageElement that contains this Node's subtree's visual form.
   * Will always be asynchronous.
   * @public
   * @deprecated - Use node.rasterized() for creating a rasterized copy, or generally it's best to get the data
   *               URL instead directly.
   *
   * @param {Function} callback - callback( image {HTMLImageElement}, x, y ) is called
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   */
  toImage: function( callback, x, y, width, height ) {

    assert && deprecationWarning( 'Node.toImage() is deprecated, please use Node.rasterized() instead' );

    assert && assert( typeof callback === 'function' );
    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    this.toDataURL( function( url, x, y ) {
      // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
      const img = document.createElement( 'img' );
      img.onload = function() {
        callback( img, x, y );
        try {
          delete img.onload;
        }
        catch( e ) {
          // do nothing
        } // fails on Safari 5.1
      };
      img.src = url;
    }, x, y, width, height );
  },

  /**
   * Calls the callback with an Image Node that contains this Node's subtree's visual form. This is always
   * asynchronous, but the resulting image Node can be used with any back-end (Canvas/WebGL/SVG/etc.)
   * @public
   * @deprecated - Use node.rasterized() instead (should avoid the asynchronous-ness)
   *
   * @param {Function} callback - callback( imageNode {Image} ) is called
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   */
  toImageNodeAsynchronous: function( callback, x, y, width, height ) {

    assert && deprecationWarning( 'Node.toImageNodeAsyncrhonous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( typeof callback === 'function' );
    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    this.toImage( function( image, x, y ) {
      callback( new Node( {
        children: [
          new scenery.Image( image, { x: -x, y: -y } )
        ]
      } ) );
    }, x, y, width, height );
  },

  /**
   * Creates a Node containing an Image Node that contains this Node's subtree's visual form. This is always
   * synchronous, but the resulting image Node can ONLY used with Canvas/WebGL (NOT SVG).
   * @public
   * @deprecated - Use node.rasterized() instead, should be mostly equivalent if useCanvas:true is provided.
   *
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   */
  toCanvasNodeSynchronous: function( x, y, width, height ) {

    assert && deprecationWarning( 'Node.toCanvasNodeSynchronous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    let result = null;
    this.toCanvas( function( canvas, x, y ) {
      result = new Node( {
        children: [
          new scenery.Image( canvas, { x: -x, y: -y } )
        ]
      } );
    }, x, y, width, height );
    assert && assert( result, 'toCanvasNodeSynchronous requires that the node can be rendered only using Canvas' );
    return result;
  },

  /**
   * Returns an Image that renders this Node. This is always synchronous, and sets initialWidth/initialHeight so that
   * we have the bounds immediately.  Use this method if you need to reduce the number of parent Nodes.
   *
   * NOTE: the resultant Image should be positioned using its bounds rather than (x,y).  To create a Node that can be
   * positioned like any other node, please use toDataURLNodeSynchronous.
   * @public
   * @deprecated - Use node.rasterized() instead, should be mostly equivalent if wrap:false is provided.
   *
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   * @returns {Image}
   */
  toDataURLImageSynchronous: function( x, y, width, height ) {

    assert && deprecationWarning( 'Node.toDataURLImageSychronous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    let result;
    this.toDataURL( function( dataURL, x, y, width, height ) {
      result = new scenery.Image( dataURL, { x: -x, y: -y, initialWidth: width, initialHeight: height } );
    }, x, y, width, height );
    assert && assert( result, 'toDataURL failed to return a result synchronously' );
    return result;
  },

  /**
   * Returns a Node that contains this Node's subtree's visual form. This is always synchronous, and sets
   * initialWidth/initialHeight so that we have the bounds immediately.  An extra wrapper Node is provided
   * so that transforms can be done independently.  Use this method if you need to be able to transform the node
   * the same way as if it had not been rasterized.
   * @public
   * @deprecated - Use node.rasterized() instead, should be mostly equivalent
   *
   * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param {number} [width] - The width of the Canvas output
   * @param {number} [height] - The height of the Canvas output
   */
  toDataURLNodeSynchronous: function( x, y, width, height ) {

    assert && deprecationWarning( 'Node.toDataURLNodeSynchronous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    return new Node( {
      children: [
        this.toDataURLImageSynchronous( x, y, width, height )
      ]
    } );
  },

  /**
   * Returns a Node (backed by a scenery Image) that is a rasterized version of this node.
   * @public
   *
   * @param {Object} [options] - See below options. This is also passed directly to the created Image object.
   * @returns {Node}
   */
  rasterized: function( options ) {
    options = merge( {
      // {number} - Controls the resolution of the image relative to the local view units. For example, if our Node is
      // ~100 view units across (in the local coordinate frame) but you want the image to actually have a ~200-pixel
      // resolution, provide resolution:2.
      resolution: 1,

      // {Bounds2|null} - If provided, it will control the x/y/width/height of the toCanvas call. See toCanvas for
      // details on how this controls the rasterization. This is in the "parent" coordinate frame, similar to
      // node.bounds.
      sourceBounds: null,

      // {boolean} - If true, the localBounds of the result will be set in a way such that it will precisely match
      // the visible bounds of the original Node (this). Note that antialiased content (with a much lower resolution)
      // may somewhat spill outside of these bounds if this is set to true. Usually this is fine and should be the
      // recommended option. If sourceBounds are provided, they will restrict the used bounds (so it will just
      // represent the bounds of the sliced part of the image).
      useTargetBounds: true,

      // {boolean} - If true, the created Image Node gets wrapped in an extra Node so that it can be transformed
      // independently. If there is no need to transform the resulting node, wrap:false can be passed so that no extra
      // Node is created.
      wrap: true,

      // {boolean} - If true, it will directly use the <canvas> element (only works with canvas/webgl renderers)
      // instead of converting this into a form that can be used with any renderer. May have slightly better
      // performance if svg/dom renderers do not need to be used.
      useCanvas: false
    }, options );

    const resolution = options.resolution;
    const sourceBounds = options.sourceBounds;

    if ( assert ) {
      assert( typeof resolution === 'number' && resolution > 0, 'resolution should be a positive number' );
      assert( sourceBounds === null || sourceBounds instanceof Bounds2, 'sourceBounds should be null or a Bounds2' );
      if ( sourceBounds ) {
        assert( sourceBounds.isValid(), 'sourceBounds should be valid (finite non-negative)' );
        assert( Utils.isInteger( sourceBounds.width ), 'sourceBounds.width should be an integer' );
        assert( Utils.isInteger( sourceBounds.height ), 'sourceBounds.height should be an integer' );
      }
    }

    // We'll need to wrap it in a container Node temporarily (while rasterizing) for the scale
    const wrapperNode = new Node( {
      scale: resolution,
      children: [ this ]
    } );

    let transformedBounds = sourceBounds || this.getSafeTransformedVisibleBounds().dilated( 2 ).roundedOut();

    // Unfortunately if we provide a resolution AND bounds, we can't use the source bounds directly.
    if ( resolution !== 1 ) {
      transformedBounds = new Bounds2(
        resolution * transformedBounds.minX,
        resolution * transformedBounds.minY,
        resolution * transformedBounds.maxX,
        resolution * transformedBounds.maxY
      );
      // Compensate for non-integral transformedBounds after our resolution transform
      if ( transformedBounds.width % 1 !== 0 ) {
        transformedBounds.maxX += 1 - ( transformedBounds.width % 1 );
      }
      if ( transformedBounds.height % 1 !== 0 ) {
        transformedBounds.maxY += 1 - ( transformedBounds.height % 1 );
      }
    }

    let image;

    // NOTE: This callback is executed SYNCHRONOUSLY
    function callback( canvas, x, y, width, height ) {
      const imageSource = options.useCanvas ? canvas : canvas.toDataURL();

      image = new scenery.Image( imageSource, merge( options, {
        x: -x,
        y: -y,
        initialWidth: width,
        initialHeight: height
      } ) );

      // We need to prepend the scale due to order of operations
      image.scale( 1 / resolution, 1 / resolution, true );
    }

    wrapperNode.toCanvas( callback, -transformedBounds.minX, -transformedBounds.minY, transformedBounds.width, transformedBounds.height );

    assert && assert( image, 'The toCanvas should have executed synchronously' );

    wrapperNode.dispose();

    // For our useTargetBounds option, we do NOT want to include any "safe" bounds, and instead want to stay true to
    // the original bounds. We do filter out invisible subtrees to set the bounds.
    let finalParentBounds = this.getVisibleBounds();
    if ( sourceBounds ) {
      // If we provide sourceBounds, don't have resulting bounds that go outside.
      finalParentBounds = sourceBounds.intersection( finalParentBounds );
    }

    if ( options.wrap ) {
      const wrappedNode = new Node( { children: [ image ] } );
      if ( options.useTargetBounds ) {
        wrappedNode.localBounds = finalParentBounds;
      }
      return wrappedNode;
    }
    else {
      if ( options.useTargetBounds ) {
        image.localBounds = image.parentToLocalBounds( finalParentBounds );
      }
      return image;
    }
  },

  /**
   * Creates a DOM drawable for this Node's self representation.
   * @public (scenery-internal)
   *
   * Implemented by subtypes that support DOM self drawables. There is no need to implement this for subtypes that
   * do not allow the DOM renderer (not set in its rendererBitmask).
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {DOMSelfDrawable}
   */
  createDOMDrawable: function( renderer, instance ) {
    throw new Error( 'createDOMDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  },

  /**
   * Creates an SVG drawable for this Node's self representation.
   * @public (scenery-internal)
   *
   * Implemented by subtypes that support SVG self drawables. There is no need to implement this for subtypes that
   * do not allow the SVG renderer (not set in its rendererBitmask).
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {SVGSelfDrawable}
   */
  createSVGDrawable: function( renderer, instance ) {
    throw new Error( 'createSVGDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  },

  /**
   * Creates a Canvas drawable for this Node's self representation.
   * @public (scenery-internal)
   *
   * Implemented by subtypes that support Canvas self drawables. There is no need to implement this for subtypes that
   * do not allow the Canvas renderer (not set in its rendererBitmask).
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {CanvasSelfDrawable}
   */
  createCanvasDrawable: function( renderer, instance ) {
    throw new Error( 'createCanvasDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  },

  /**
   * Creates a WebGL drawable for this Node's self representation.
   * @public (scenery-internal)
   *
   * Implemented by subtypes that support WebGL self drawables. There is no need to implement this for subtypes that
   * do not allow the WebGL renderer (not set in its rendererBitmask).
   *
   * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param {Instance} instance - Instance object that will be associated with the drawable
   * @returns {WebGLSelfDrawable}
   */
  createWebGLDrawable: function( renderer, instance ) {
    throw new Error( 'createWebGLDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  },

  /*---------------------------------------------------------------------------*
   * Instance handling
   *----------------------------------------------------------------------------*/

  /**
   * Returns a reference to the instances array.
   * @public (scenery-internal)
   *
   * @returns {Array.<Instance>}
   */
  getInstances: function() {
    return this._instances;
  },
  get instances() { return this.getInstances(); },

  /**
   * Adds an Instance reference to our array.
   * @public (scenery-internal)
   *
   * @param {Instance} instance
   */
  addInstance: function( instance ) {
    assert && assert( instance instanceof Instance );
    this._instances.push( instance );

    this.changedInstanceEmitter.emit( instance, true );
  },

  /**
   * Removes an Instance reference from our array.
   * @public (scenery-internal)
   *
   * @param {Instance} instance
   */
  removeInstance: function( instance ) {
    assert && assert( instance instanceof Instance );
    const index = _.indexOf( this._instances, instance );
    assert && assert( index !== -1, 'Cannot remove a Instance from a Node if it was not there' );
    this._instances.splice( index, 1 );

    this.changedInstanceEmitter.emit( instance, false );
  },

  /**
   * Returns whether this Node was visually rendered/displayed by a Display in the last updateDisplay() call.
   * @public
   *
   * @param {Display} display
   * @returns {boolean}
   */
  wasDisplayed: function( display ) {
    for ( let i = 0; i < this._instances.length; i++ ) {
      const instance = this._instances[ i ];

      if ( instance.display === display && instance.visible ) {
        return true;
      }
    }
    return false;
  },

  /*---------------------------------------------------------------------------*
   * Display handling
   *----------------------------------------------------------------------------*/

  /**
   * Returns a reference to the display array.
   * @public (scenery-internal)
   *
   * @returns {Array.<Display>}
   */
  getRootedDisplays: function() {
    return this._rootedDisplays;
  },
  get rootedDisplays() { return this.getRootedDisplays(); },

  /**
   * Adds an display reference to our array.
   * @public (scenery-internal)
   *
   * @param {Display} display
   */
  addRootedDisplay: function( display ) {
    assert && assert( display instanceof scenery.Display );
    this._rootedDisplays.push( display );

    // Defined in ParallelDOM.js
    this._accessibleDisplaysInfo.onAddedRootedDisplay( display );
  },

  /**
   * Removes a Display reference from our array.
   * @public (scenery-internal)
   *
   * @param {Display} display
   */
  removeRootedDisplay: function( display ) {
    assert && assert( display instanceof scenery.Display );
    const index = _.indexOf( this._rootedDisplays, display );
    assert && assert( index !== -1, 'Cannot remove a Display from a Node if it was not there' );
    this._rootedDisplays.splice( index, 1 );

    // Defined in ParallelDOM.js
    this._accessibleDisplaysInfo.onRemovedRootedDisplay( display );
  },

  /*---------------------------------------------------------------------------*
   * Coordinate transform methods
   *----------------------------------------------------------------------------*/

  /**
   * Returns a point transformed from our local coordinate frame into our parent coordinate frame. Applies our node's
   * transform to it.
   * @public
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  localToParentPoint: function( point ) {
    return this._transform.transformPosition2( point );
  },

  /**
   * Returns bounds transformed from our local coordinate frame into our parent coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  localToParentBounds: function( bounds ) {
    return this._transform.transformBounds2( bounds );
  },

  /**
   * Returns a point transformed from our parent coordinate frame into our local coordinate frame. Applies the inverse
   * of our node's transform to it.
   * @public
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  parentToLocalPoint: function( point ) {
    return this._transform.inversePosition2( point );
  },

  /**
   * Returns bounds transformed from our parent coordinate frame into our local coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  parentToLocalBounds: function( bounds ) {
    return this._transform.inverseBounds2( bounds );
  },

  /**
   * A mutable-optimized form of localToParentBounds() that will modify the provided bounds, transforming it from our
   * local coordinate frame to our parent coordinate frame.
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2} - The same bounds object.
   */
  transformBoundsFromLocalToParent: function( bounds ) {
    return bounds.transform( this._transform.getMatrix() );
  },

  /**
   * A mutable-optimized form of parentToLocalBounds() that will modify the provided bounds, transforming it from our
   * parent coordinate frame to our local coordinate frame.
   * @public
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2} - The same bounds object.
   */
  transformBoundsFromParentToLocal: function( bounds ) {
    return bounds.transform( this._transform.getInverse() );
  },

  /**
   * Returns a new matrix (fresh copy) that would transform points from our local coordinate frame to the global
   * coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @returns {Matrix3}
   */
  getLocalToGlobalMatrix: function() {
    let node = this; // eslint-disable-line consistent-this

    // we need to apply the transformations in the reverse order, so we temporarily store them
    const matrices = [];

    // concatenation like this has been faster than getting a unique trail, getting its transform, and applying it
    while ( node ) {
      matrices.push( node._transform.getMatrix() );
      assert && assert( node._parents[ 1 ] === undefined, 'getLocalToGlobalMatrix unable to work for DAG' );
      node = node._parents[ 0 ];
    }

    const matrix = Matrix3.identity(); // will be modified in place

    // iterate from the back forwards (from the root Node to here)
    for ( let i = matrices.length - 1; i >= 0; i-- ) {
      matrix.multiplyMatrix( matrices[ i ] );
    }

    // NOTE: always return a fresh copy, getGlobalToLocalMatrix depends on it to minimize instance usage!
    return matrix;
  },

  /**
   * Returns a Transform3 that would transform things from our local coordinate frame to the global coordinate frame.
   * Equivalent to getUniqueTrail().getTransform(), but faster.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @returns {Transform3}
   */
  getUniqueTransform: function() {
    return new Transform3( this.getLocalToGlobalMatrix() );
  },

  /**
   * Returns a new matrix (fresh copy) that would transform points from the global coordinate frame to our local
   * coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @returns {Matrix3}
   */
  getGlobalToLocalMatrix: function() {
    return this.getLocalToGlobalMatrix().invert();
  },

  /**
   * Transforms a point from our local coordinate frame to the global coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  localToGlobalPoint: function( point ) {
    let node = this; // eslint-disable-line consistent-this
    const resultPoint = point.copy();
    while ( node ) {
      // in-place multiplication
      node._transform.getMatrix().multiplyVector2( resultPoint );
      assert && assert( node._parents[ 1 ] === undefined, 'localToGlobalPoint unable to work for DAG' );
      node = node._parents[ 0 ];
    }
    return resultPoint;
  },

  /**
   * Transforms a point from the global coordinate frame to our local coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  globalToLocalPoint: function( point ) {
    let node = this; // eslint-disable-line consistent-this
    // TODO: performance: test whether it is faster to get a total transform and then invert (won't compute individual inverses)

    // we need to apply the transformations in the reverse order, so we temporarily store them
    const transforms = [];
    while ( node ) {
      transforms.push( node._transform );
      assert && assert( node._parents[ 1 ] === undefined, 'globalToLocalPoint unable to work for DAG' );
      node = node._parents[ 0 ];
    }

    // iterate from the back forwards (from the root Node to here)
    const resultPoint = point.copy();
    for ( let i = transforms.length - 1; i >= 0; i-- ) {
      // in-place multiplication
      transforms[ i ].getInverse().multiplyVector2( resultPoint );
    }
    return resultPoint;
  },

  /**
   * Transforms bounds from our local coordinate frame to the global coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  localToGlobalBounds: function( bounds ) {
    // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
    // it also seems to be a bit faster this way
    return bounds.transformed( this.getLocalToGlobalMatrix() );
  },

  /**
   * Transforms bounds from the global coordinate frame to our local coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  globalToLocalBounds: function( bounds ) {
    // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
    return bounds.transformed( this.getGlobalToLocalMatrix() );
  },

  /**
   * Transforms a point from our parent coordinate frame to the global coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  parentToGlobalPoint: function( point ) {
    assert && assert( this.parents.length <= 1, 'parentToGlobalPoint unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].localToGlobalPoint( point ) : point;
  },

  /**
   * Transforms bounds from our parent coordinate frame to the global coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  parentToGlobalBounds: function( bounds ) {
    assert && assert( this.parents.length <= 1, 'parentToGlobalBounds unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].localToGlobalBounds( bounds ) : bounds;
  },

  /**
   * Transforms a point from the global coordinate frame to our parent coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Vector2} point
   * @returns {Vector2}
   */
  globalToParentPoint: function( point ) {
    assert && assert( this.parents.length <= 1, 'globalToParentPoint unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].globalToLocalPoint( point ) : point;
  },

  /**
   * Transforms bounds from the global coordinate frame to our parent coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * @param {Bounds2} bounds
   * @returns {Bounds2}
   */
  globalToParentBounds: function( bounds ) {
    assert && assert( this.parents.length <= 1, 'globalToParentBounds unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].globalToLocalBounds( bounds ) : bounds;
  },

  /**
   * Returns a bounding box for this Node (and its sub-tree) in the global coordinate frame.
   * @public
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * NOTE: This requires computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @returns {Bounds2}
   */
  getGlobalBounds: function() {
    assert && assert( this.parents.length <= 1, 'globalBounds unable to work for DAG' );
    return this.parentToGlobalBounds( this.getBounds() );
  },
  get globalBounds() { return this.getGlobalBounds(); },

  /**
   * Returns the bounds of any other Node in our local coordinate frame.
   *
   * NOTE: If this node or the passed in Node have multiple instances (e.g. this or one ancestor has two parents), it will fail
   * with an assertion.
   *
   * TODO: Possible to be well-defined and have multiple instances of each.
   *
   * @param {Node} node
   * @returns {Bounds2}
   */
  boundsOf: function( node ) {
    return this.globalToLocalBounds( node.getGlobalBounds() );
  },

  /**
   * Returns the bounds of this Node in another node's local coordinate frame.
   *
   * NOTE: If this node or the passed in Node have multiple instances (e.g. this or one ancestor has two parents), it will fail
   * with an assertion.
   *
   * TODO: Possible to be well-defined and have multiple instances of each.
   *
   * @param {Node} node
   * @returns {Bounds2}
   */
  boundsTo: function( node ) {
    return node.globalToLocalBounds( this.getGlobalBounds() );
  },

  /*---------------------------------------------------------------------------*
   * Drawable handling
   *----------------------------------------------------------------------------*/

  /**
   * Adds the drawable to our list of drawables to notify of visual changes.
   * @public (scenery-internal)
   *
   * @param {Drawable} drawable
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  attachDrawable: function( drawable ) {
    this._drawables.push( drawable );
    return this; // allow chaining
  },

  /**
   * Removes the drawable from our list of drawables to notify of visual changes.
   * @public (scenery-internal)
   *
   * @param {Drawable} drawable
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  detachDrawable: function( drawable ) {
    const index = _.indexOf( this._drawables, drawable );

    assert && assert( index >= 0, 'Invalid operation: trying to detach a non-referenced drawable' );

    this._drawables.splice( index, 1 ); // TODO: replace with a remove() function
    return this;
  },

  /**
   * Scans the options object for key names that correspond to ES5 setters or other setter functions, and calls those
   * with the values.
   * @public
   *
   * For example:
   *
   * node.mutate( { top: 0, left: 5 } );
   *
   * will be equivalent to:
   *
   * node.left = 5;
   * node.top = 0;
   *
   * In particular, note that the order is different. Mutators will be applied in the order of _mutatorKeys, which can
   * be added to by subtypes.
   *
   * Additionally, some keys are actually direct function names, like 'scale'. mutate( { scale: 2 } ) will call
   * node.scale( 2 ) instead of activating an ES5 setter directly.
   *
   * @param {Object} [options]
   * @returns {Node} - Returns 'this' reference, for chaining
   */
  mutate: function( options ) {

    if ( !options ) {
      return this;
    }

    assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    assert && assert( _.filter( [ 'translation', 'x', 'left', 'right', 'centerX', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom' ], function( key ) { return options[ key ] !== undefined; } ).length <= 1,
      'More than one mutation on this Node set the x component, check ' + Object.keys( options ).join( ',' ) );

    assert && assert( _.filter( [ 'translation', 'y', 'top', 'bottom', 'centerY', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom' ], function( key ) { return options[ key ] !== undefined; } ).length <= 1,
      'More than one mutation on this Node set the y component, check ' + Object.keys( options ).join( ',' ) );

    const self = this;

    _.each( this._mutatorKeys, function( key ) {

      // See https://github.com/phetsims/scenery/issues/580 for more about passing undefined.
      assert && assert( !options.hasOwnProperty( key ) || options[ key ] !== undefined, 'Undefined not allowed for Node key: ' + key );

      if ( options[ key ] !== undefined ) {
        const descriptor = Object.getOwnPropertyDescriptor( Node.prototype, key );

        // if the key refers to a function that is not ES5 writable, it will execute that function with the single argument
        if ( descriptor && typeof descriptor.value === 'function' ) {
          self[ key ]( options[ key ] );
        }
        else {
          self[ key ] = options[ key ];
        }
      }
    } );

    this.initializePhetioObject( { phetioType: Node.NodeIO, phetioState: false }, options );

    return this; // allow chaining
  },

  /**
   * @param {Object} baseOptions
   * @param {Object} config
   * @override
   * @protected
   */
  initializePhetioObject: function( baseOptions, config ) {

    config = merge( {

      // This option is used to create the instrumented, default PhET-iO visibleProperty. These options should not
      // be provided if a `visibleProperty` was provided to this Node, though if they are, they will just be ignored.
      // This grace is to support default options across the component hierarchy melding with usages providing a visibleProperty.
      // This option is a bit buried because it can only be used when the Node is being instrumented, which is when
      // the default, instrumented visibleProperty is conditionally created. We don't want to store these on the Node,
      // and thus they aren't support through `mutate()`.
      visiblePropertyOptions: null,
      pickablePropertyOptions: null
    }, config );

    // Track this, so we only override our visibleProperty once.
    const wasInstrumented = this.isPhetioInstrumented();

    PhetioObject.prototype.initializePhetioObject.call( this, baseOptions, config );

    if ( !wasInstrumented && this.isPhetioInstrumented() ) {

      // For each supported TinyForwardingProperty, if a Property was already specified in the options (in the
      // constructor or mutate), then it will be set as this.targetProperty there. Here we only create the default
      // instrumented one if another hasn't already been specified.

      this._visibleProperty.initializePhetio( this, VISIBLE_PROPERTY_TANDEM_NAME, () => new BooleanProperty( this.visible, merge( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          tandem: this.tandem.createTandem( VISIBLE_PROPERTY_TANDEM_NAME ),
          phetioDocumentation: 'Controls whether the Node will be visible (and interactive), see the NodeIO documentation for more details.'
        }, config.visiblePropertyOptions ) )
      );

      this._pickableProperty.initializePhetio( this, PICKABLE_PROPERTY_TANDEM_NAME, () => new Property( this.pickable, merge( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          tandem: this.tandem.createTandem( PICKABLE_PROPERTY_TANDEM_NAME ),
          phetioType: Property.PropertyIO( NullableIO( BooleanIO ) ),
          phetioFeatured: true, // Since this property is opt-in, we typically only opt-in when it should be featured
          phetioDocumentation: 'Sets whether the node will be pickable (and hence interactive). Pickable can take one ' +
                               'of three values:<br>' +
                               '<ul>' +
                               '<li>null: pass-through behavior. Nodes with input listeners are pickable, but nodes without input listeners won\'t block events for nodes behind it.</li>' +
                               '<li>false: The node cannot be interacted with, and it blocks events for nodes behind it.</li>' +
                               '<li>true: The node can be interacted with (if it has an input listener).</li>' +
                               '</ul>' +
                               'For more about Scenery node pickability, please see <a href="http://phetsims.github.io/scenery/doc/implementation-notes#pickability">http://phetsims.github.io/scenery/doc/implementation-notes#pickability</a>'
        }, config.pickablePropertyOptions ) )
      );
    }
  },

  /**
   * Override for extra information in the debugging output (from Display.getDebugHTML()).
   * @protected (scenery-internal)
   *
   * @returns {string}
   */
  getDebugHTMLExtras: function() {
    return '';
  },

  /**
   * Makes this Node's subtree available for inspection.
   * @public
   */
  inspect: function() {
    localStorage.scenerySnapshot = JSON.stringify( {
      type: 'Subtree',
      rootNodeId: this.id,
      nodes: scenery.serializeConnectedNodes( this )
    } );
  },

  /**
   * Returns a debugging string that is an attempted serialization of this node's sub-tree.
   * @public
   *
   * @param {string} spaces - Whitespace to add
   * @param {boolean} [includeChildren]
   */
  toString: function( spaces, includeChildren ) {
    return this.constructor.name + '#' + this.id;
  },

  /**
   * Performs checks to see if the internal state of Instance references is correct at a certain point in/after the
   * Display's updateDisplay().
   * @private
   */
  auditInstanceSubtreeForDisplay: function( display ) {
    if ( assertSlow ) {
      const numInstances = this._instances.length;
      for ( let i = 0; i < numInstances; i++ ) {
        const instance = this._instances[ i ];
        if ( instance.display === display ) {
          assertSlow( instance.trail.isValid(),
            'Invalid trail on Instance: ' + instance.toString() + ' with trail ' + instance.trail.toString() );
        }
      }

      // audit all of the children
      this.children.forEach( function( child ) {
        child.auditInstanceSubtreeForDisplay( display );
      } );
    }
  },

  /**
   * When we add or remove any number of bounds listeners, we want to increment/decrement internal information.
   * @private
   *
   * @param {number} deltaQuantity - If positive, the number of listeners being added, otherwise the number removed
   */
  onBoundsListenersAddedOrRemoved: function( deltaQuantity ) {
    this.changeBoundsEventCount( deltaQuantity );
    this._boundsEventSelfCount += deltaQuantity;
  },

  /**
   * Disposes the node, releasing all references that it maintained.
   * @public
   */
  dispose: function() {

    // remove all accessibility input listeners
    this.disposeAccessibility();

    // When disposing, remove all children and parents. See https://github.com/phetsims/scenery/issues/629
    this.removeAllChildren();
    this.detach();

    this._visibleProperty.dispose();
    this._pickableProperty.dispose();

    // Tear-down in the reverse order Node was created
    PhetioObject.prototype.dispose.call( this );
  },

  /**
   * Disposes this Node and all other descendant nodes.
   * @public
   *
   * NOTE: Use with caution, as you should not re-use any Node touched by this. Not compatible with most DAG
   *       techniques.
   */
  disposeSubtree: function() {
    if ( !this.isDisposed ) {
      // makes a copy before disposing
      const children = this.children;

      this.dispose();

      for ( let i = 0; i < children.length; i++ ) {
        children[ i ].disposeSubtree();
      }
    }
  }
}, {
  // @public {Object} - A mapping of all of the default options provided to Node
  DEFAULT_OPTIONS: DEFAULT_OPTIONS,

  /**
   * A default for getTrails() searches, returns whether the Node has no parents.
   * @public
   *
   * @param {Node} node
   * @returns {boolean}
   */
  defaultTrailPredicate: function( node ) {
    return node._parents.length === 0;
  },

  defaultLeafTrailPredicate: function( node ) {
    return node._children.length === 0;
  }
} );

// Node is composed with this feature of Interactive Descriptions
ParallelDOM.compose( Node );

Node.NodeIO = new IOType( 'NodeIO', {
  valueType: Node,
  documentation: 'The base type for graphical and potentially interactive objects.'
} );

export default Node;