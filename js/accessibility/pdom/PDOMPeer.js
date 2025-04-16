// Copyright 2015-2025, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM. A PDOMPeer can
 * have up to four window.Elements displayed in the PDOM, see ftructor for details.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

import Bounds2 from '../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../dot/js/Matrix3.js';
import arrayRemove from '../../../../phet-core/js/arrayRemove.js';
import merge from '../../../../phet-core/js/merge.js';
import platform from '../../../../phet-core/js/platform.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import FocusManager from '../../accessibility/FocusManager.js';
import PDOMSiblingStyle from '../../accessibility/pdom/PDOMSiblingStyle.js';
import PDOMUtils from '../../accessibility/pdom/PDOMUtils.js';
import scenery from '../../scenery.js';
import { pdomFocusProperty } from '../pdomFocusProperty.js';
import { guessVisualTrail } from './guessVisualTrail.js';
import { PEER_ACCESSIBLE_PARAGRAPH_SIBLING } from './PEER_ACCESSIBLE_PARAGRAPH_SIBLING.js';
import { PEER_CONTAINER_PARENT } from './PEER_CONTAINER_PARENT.js';
import { PEER_DESCRIPTION_SIBLING } from './PEER_DESCRIPTION_SIBLING.js';
import { PEER_HEADING_SIBLING } from './PEER_HEADING_SIBLING.js';
import { PEER_LABEL_SIBLING } from './PEER_LABEL_SIBLING.js';
import { PEER_PRIMARY_SIBLING } from './PEER_PRIMARY_SIBLING.js';

// constants
const PRIMARY_SIBLING = PEER_PRIMARY_SIBLING;
const HEADING_SIBLING = PEER_HEADING_SIBLING;
const LABEL_SIBLING = PEER_LABEL_SIBLING;
const DESCRIPTION_SIBLING = PEER_DESCRIPTION_SIBLING;
const ACCESSIBLE_PARAGRAPH_SIBLING = PEER_ACCESSIBLE_PARAGRAPH_SIBLING;
const CONTAINER_PARENT = PEER_CONTAINER_PARENT;
const LABEL_TAG = PDOMUtils.TAGS.LABEL;
const INPUT_TAG = PDOMUtils.TAGS.INPUT;
const DISABLED_ATTRIBUTE_NAME = 'disabled';

// DOM observers that apply new CSS transformations are triggered when children, or inner content change. Updating
// style/positioning of the element will change attributes so we can't observe those changes since it would trigger
// the MutationObserver infinitely.
const OBSERVER_CONFIG = { attributes: false, childList: true, characterData: true };

let globalId = 1;

// mutables instances to avoid creating many in operations that occur frequently
const scratchGlobalBounds = new Bounds2( 0, 0, 0, 0 );
const scratchSiblingBounds = new Bounds2( 0, 0, 0, 0 );
const globalNodeTranslationMatrix = new Matrix3();
const globalToClientScaleMatrix = new Matrix3();
const nodeScaleMagnitudeMatrix = new Matrix3();

class PDOMPeer {
  /**
   * @param {PDOMInstance} pdomInstance
   * @param {Object} [options]
   * @mixes Poolable
   */
  constructor( pdomInstance, options ) {
    this.initializePDOMPeer( pdomInstance, options );
  }

  /**
   * Initializes the object (either from a freshly-created state, or from a "disposed" state brought back from a
   * pool).
   *
   * NOTE: the PDOMPeer is not fully constructed until calling PDOMPeer.update() after creating from pool.
   * @private
   *
   * @param {PDOMInstance} pdomInstance
   * @param {Object} [options]
   * @returns {PDOMPeer} - Returns 'this' reference, for chaining
   */
  initializePDOMPeer( pdomInstance, options ) {
    options = merge( {
      primarySibling: null
    }, options );

    assert && assert( !this.id || this.isDisposed, 'If we previously existed, we need to have been disposed' );

    // @public {number} - unique ID
    this.id = this.id || globalId++;

    // @public {PDOMInstance}
    this.pdomInstance = pdomInstance;

    // @public {Node|null} only null for the root pdomInstance
    this.node = this.pdomInstance.node;

    // @public {Display} - Each peer is associated with a specific Display.
    this.display = pdomInstance.display;

    // @public {Trail} - NOTE: May have "gaps" due to pdomOrder usage.
    this.trail = pdomInstance.trail;

    // @private {boolean|null} - whether or not this PDOMPeer is visible in the PDOM
    // Only initialized to null, should not be set to it. isVisible() will return true if this.visible is null
    // (because it hasn't been set yet).
    this.visible = null;

    // @private {boolean|null} - whether or not the primary sibling of this PDOMPeer can receive focus.
    this.focusable = null;

    // @private {HTMLElement|null} - Optional label/description elements
    this._headingSibling = null;
    this._labelSibling = null;
    this._descriptionSibling = null;

    // @private {HTMLElement|null} - Optional paragraph element for the "high level" API.
    this._accessibleParagraphSibling = null;

    // @private {HTMLElement|null} - A parent element that can contain this primarySibling and other siblings, usually
    // the label and description content.
    this._containerParent = null;

    // @public {HTMLElement[]} Rather than guarantee that a peer is a tree with a root DOMElement,
    // allow multiple window.Elements at the top level of the peer. This is used for sorting the instance.
    // See this.orderElements for more info.
    this.topLevelElements = [];

    // @private {boolean} - flag that indicates that this peer has accessible content that changed, and so
    // the siblings need to be repositioned in the next Display.updateDisplay()
    this.positionDirty = false;

    // @private {boolean} - Flag that indicates that PDOM elements require a forced reflow next animation frame.
    // This is needed to fix a Safari VoiceOver bug where the accessible name is read incorrectly after elements
    // are hidden/displayed. The usual workaround to force a reflow (set the style.display to none, query the offset,
    // set it back) only fixes the problem if the style.display attribute is set in the next animation frame.
    // See https://github.com/phetsims/a11y-research/issues/193.
    this.forceReflowWorkaround = false;

    // @private {boolean} - indicates that this peer's pdomInstance has a descendant that is dirty. Used to
    // quickly find peers with positionDirty when we traverse the tree of PDOMInstances
    this.childPositionDirty = false;

    // @private {boolean} - Indicates that this peer will position sibling elements so that
    // they are in the right location in the viewport, which is a requirement for touch based
    // screen readers. See setPositionInPDOM.
    this.positionInPDOM = false;

    // @private {MutationObserver} - An observer that will call back any time a property of the primary
    // sibling changes. Used to reposition the sibling elements if the bounding box resizes. No need to loop over
    // all of the mutations, any single mutation will require updating CSS positioning.
    //
    // NOTE: Ideally, a single MutationObserver could be used to observe changes to all elements in the PDOM. But
    // MutationObserver makes it impossible to detach observers from a single element. MutationObserver.detach()
    // will remove listeners on all observed elements, so individual observers must be used on each element.
    // One alternative could be to put the MutationObserver on the root element and use "subtree: true" in
    // OBSERVER_CONFIG. This could reduce the number of MutationObservers, but there is no easy way to get the
    // peer from the mutation target element. If MutationObserver takes a lot of memory, this could be an
    // optimization that may come with a performance cost.
    //
    // NOTE: ResizeObserver is a superior alternative to MutationObserver for this purpose because
    // it will only monitor changes we care about and prevent infinite callback loops if size is changed in
    // the callback function (we get around this now by not observing attribute changes). But it is not yet widely
    // supported, see https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver.
    //
    // TODO: Should we be watching "model" changes from ParallelDOM.js instead of using MutationObserver? https://github.com/phetsims/scenery/issues/1581
    // See https://github.com/phetsims/scenery/issues/852. This would be less fragile, and also less
    // memory intensive because we don't need an instance of MutationObserver on every PDOMInstance.
    this.mutationObserver = this.mutationObserver || new MutationObserver( this.invalidateCSSPositioning.bind( this, false ) );

    // @private {function} - must be removed on disposal
    this.transformListener = this.transformListener || this.invalidateCSSPositioning.bind( this, false );
    this.pdomInstance.transformTracker.addListener( this.transformListener );

    // @private {*} - To support setting the Display.interactive=false (which sets disabled on all primarySiblings,
    // we need to set disabled on a separate channel from this.setAttributeToElement. That way we cover the case where
    // `disabled` was set through the ParallelDOM API when we need to toggle it specifically for Display.interactive.
    // This way we can conserve the previous `disabled` attribute/property value through toggling Display.interactive.
    this._preservedDisabledValue = null;

    // @private {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
    // interacted with.
    this.isDisposed = false;

    // edge case for root accessibility
    if ( this.pdomInstance.isRootInstance ) {

      // @private {HTMLElement} - The main element associated with this peer. If focusable, this is the element that gets
      // the focus. It also will contain any children.
      this._primarySibling = options.primarySibling;
      this._primarySibling.classList.add( PDOMSiblingStyle.ROOT_CLASS_NAME );

      // Stop blocked events from bubbling past the root of the PDOM so that scenery does
      // not dispatch them in Input.js.
      PDOMUtils.BLOCKED_DOM_EVENTS.forEach( eventType => {
        this._primarySibling.addEventListener( eventType, event => {
          event.stopPropagation();
        } );
      } );
    }

    return this;
  }

  /**
   * Update the content of the peer. This must be called after the AccessiblePeer is constructed from pool.
   * @param {boolean} updateIndicesStringAndElementIds - if this function should be called upon initial "construction" (in update), allows for the option to do this lazily, see https://github.com/phetsims/phet-io/issues/1847
   * @public (scenery-internal)
   */
  update( updateIndicesStringAndElementIds ) {
    // NOTE: JO: 2025-04-01: This looks like it will usually convert string Properties to strings (because of the getters)
    // but this will probably make TypeScript conversion problematic.
    let options = this.node.getBaseOptions();

    const callbacksForOtherNodes = [];

    // In case update() is called more than once on an instance of PDOMPeer. Observer will be set up
    // later if there is a primary sibling.
    this.mutationObserver.disconnect();

    // Even if the accessibleName is null, we need to run the behavior function if the dirty flag is set
    // to run any cleanup on Nodes changed with callbacksForOtherNodes. See https://github.com/phetsims/scenery/issues/1679.
    if ( this.node.accessibleName !== null || this.node._accessibleNameDirty ) {
      if ( this.node.accessibleName === null ) {

        // There is no accessibleName, so we don't want to modify options - just run the behavior for cleanup
        // and to update other nodes.
        this.node.accessibleNameBehavior( this.node, {}, this.node.accessibleName, callbacksForOtherNodes );
      }
      else {
        options = this.node.accessibleNameBehavior( this.node, options, this.node.accessibleName, callbacksForOtherNodes );
      }
      assert && assert( typeof options === 'object', 'should return an object' );
      this.node._accessibleNameDirty = false;
    }

    // Even if the accessibleHelpText is null, we need to run the behavior function if the dirty flag is set
    // to run any cleanup on Nodes changed with callbacksForOtherNodes. See https://github.com/phetsims/scenery/issues/1679.
    if ( this.node.accessibleHelpText !== null || this.node._accessibleHelpTextDirty ) {
      if ( this.node.accessibleHelpText === null ) {

        // There is no accessibleHelpText, so we don't want to modify options - just run the behavior for cleanup
        // and to update other nodes.
        this.node.accessibleHelpTextBehavior( this.node, {}, this.node.accessibleHelpText, callbacksForOtherNodes );
      }
      else {
        options = this.node.accessibleHelpTextBehavior( this.node, options, this.node.accessibleHelpText, callbacksForOtherNodes );
      }
      assert && assert( typeof options === 'object', 'should return an object' );
      this.node._accessibleHelpTextDirty = false;
    }

    // Even if the accessibleHelpText is null, we need to run the behavior function if the dirty flag is set
    // to run any cleanup on Nodes changed with callbacksForOtherNodes. See https://github.com/phetsims/scenery/issues/1679.
    if ( this.node.accessibleParagraph !== null || this.node._accessibleParagraphDirty ) {
      if ( this.node.accessibleParagraph === null ) {

        // There is no accessibleParagraph, so we don't want to modify options - just run the behavior for cleanup
        // and to update other nodes.
        this.node.accessibleParagraphBehavior( this.node, {}, this.node.accessibleParagraph, callbacksForOtherNodes );
      }
      else {
        options = this.node.accessibleParagraphBehavior( this.node, options, this.node.accessibleParagraph, callbacksForOtherNodes );
      }
      assert && assert( typeof options === 'object', 'should return an object' );
      this.node._accessibleParagraphDirty = false;
    }

    // If there is an accessible paragraph, create it now. Accessible paragraph does not require a tagName. But assertions are thrown
    // if this instance has children without a tagName, so one is not created by default.
    if ( options.accessibleParagraph ) {
      this._accessibleParagraphSibling = createElement( PDOMUtils.TAGS.P, false );
      this.setAccessibleParagraphContent( options.accessibleParagraph );
    }

    // accessibleHeading can be used without a tagName, and it enables PDOM for the Node. If there is no tagName, create one
    // so that children under the heading are added under it by default.
    // This is done in PDOMPeer instead of on the Node, so that removing accessibleHeading from the Node when there is no
    // set tagName will fully remove all content from the DOM (clear all PDOMInstances).
    if ( options.accessibleHeading && !options.tagName ) {
      options.tagName = PDOMUtils.TAGS.DIV;
    }

    // create the base DOM element representing this accessible instance
    if ( options.tagName ) {

      // TODO: why not just options.focusable? https://github.com/phetsims/scenery/issues/1581
      this._primarySibling = createElement( options.tagName, this.node.focusable, {
        namespace: options.pdomNamespace
      } );

      // create the container parent for the dom siblings
      if ( options.containerTagName ) {
        this._containerParent = createElement( options.containerTagName, false );
      }

      if ( options.accessibleHeading ) {
        this._headingSibling = createElement( `h${this.pdomInstance.pendingHeadingLevel}`, false );
      }

      // create the label DOM element representing this instance
      if ( options.labelTagName ) {
        this._labelSibling = createElement( options.labelTagName, false, {
          excludeFromInput: this.node._excludeLabelSiblingFromInput
        } );
      }

      // create the description DOM element representing this instance
      if ( options.descriptionTagName ) {
        this._descriptionSibling = createElement( options.descriptionTagName, false );
      }
    }

    // Set ids on the elements - Attributes set below this call may require that the ids are set!!
    updateIndicesStringAndElementIds && this.updateIndicesStringAndElementIds();

    this.orderElements( options );

    // The primary sibling (set with Node.tagName) is required for the peer to be visible in the PDOM.
    if ( this._primarySibling ) {

      // assign listeners (to be removed or disconnected during disposal)
      this.mutationObserver.observe( this._primarySibling, OBSERVER_CONFIG );

      if ( options.accessibleHeading !== null ) {
        this.setHeadingContent( options.accessibleHeading );
      }

      // set the accessible label now that the element has been recreated again, but not if the tagName
      // has been cleared out
      if ( options.labelContent && options.labelTagName !== null ) {
        this.setLabelSiblingContent( options.labelContent );
      }

      // restore the innerContent
      if ( options.innerContent && options.tagName !== null ) {
        this.setPrimarySiblingContent( options.innerContent );
      }

      // set the accessible description, but not if the tagName has been cleared out.
      if ( options.descriptionContent && options.descriptionTagName !== null ) {
        this.setDescriptionSiblingContent( options.descriptionContent );
      }

      // if element is an input element, set input type
      if ( options.tagName.toUpperCase() === INPUT_TAG && options.inputType ) {
        this.setAttributeToElement( 'type', options.inputType );
      }

      // if the label element happens to be a 'label', associate with 'for' attribute (must be done after updating IDs)
      if ( options.labelTagName && options.labelTagName.toUpperCase() === LABEL_TAG ) {
        this.setAttributeToElement( 'for', this._primarySibling.id, {
          elementName: PDOMPeer.LABEL_SIBLING
        } );
      }

      this.setFocusable( this.node.focusable );

      // set the positionInPDOM field to our updated instance
      this.setPositionInPDOM( this.node.positionInPDOM );

      // recompute and assign the association attributes that link two elements (like aria-labelledby)
      this.onAriaLabelledbyAssociationChange();
      this.onAriaDescribedbyAssociationChange();
      this.onActiveDescendantAssociationChange();

      // update all attributes for the peer, should cover aria-label, role, and others
      this.onAttributeChange( options );

      // update all classes for the peer
      this.onClassChange();

      // update input value attribute for the peer
      this.onInputValueChange();

      this.node.updateOtherNodesAriaLabelledby();
      this.node.updateOtherNodesAriaDescribedby();
      this.node.updateOtherNodesActiveDescendant();
    }

    callbacksForOtherNodes.forEach( callback => {
      assert && assert( typeof callback === 'function' );
      callback();
    } );
  }

  /**
   * Handle the internal ordering of the elements in the peer, this involves setting the proper value of
   * this.topLevelElements
   * @param {Object} config - the computed mixin options to be applied to the peer. (select ParallelDOM mutator keys)
   * @private
   */
  orderElements( config ) {
    if ( this._containerParent ) {
      // The first child of the container parent element should be the peer dom element
      // if undefined, the insertBefore method will insert the this._primarySibling as the first child
      assert && assert( this._primarySibling, 'There should be a _primarySibling if there is a _containerParent' );
      this._containerParent.insertBefore( this._primarySibling, this._containerParent.children[ 0 ] || null );
      this.topLevelElements = [ this._containerParent ];
    }
    else {

      // Wean out any null siblings
      this.topLevelElements = [ this._headingSibling, this._labelSibling, this._descriptionSibling, this._primarySibling, this._accessibleParagraphSibling ].filter( _.identity );
    }

    // insert the heading/label/description elements in the correct location if they exist
    // NOTE: Important for arrangeContentElement to be called on the heading sibling, then the label sibling first for
    // correct order. This should ensure that the heading sibling is first (if present)
    this._headingSibling && this.arrangeContentElement( this._headingSibling, false );
    this._labelSibling && this.arrangeContentElement( this._labelSibling, config.appendLabel );
    this._descriptionSibling && this.arrangeContentElement( this._descriptionSibling, config.appendDescription );
    this._accessibleParagraphSibling && this.arrangeContentElement( this._accessibleParagraphSibling, true );
  }

  /**
   * Returns the sibling that can be placed in the PDOM. This is the primary sibling if it exists, otherwise the
   * accessible paragraph.
   *
   * If other elements can be placed without the primary sibling, they could be added to this function.
   *
   * @public
   */
  getPlaceableSibling() {
    const placeable = this._primarySibling || this._accessibleParagraphSibling;
    assert && assert( placeable, 'No placeable sibling found!' );

    return placeable;
  }

  /**
   * Get the primary sibling element for the peer
   * @public
   * @returns {HTMLElement|null}
   */
  getPrimarySibling() {
    return this._primarySibling;
  }

  get primarySibling() { return this.getPrimarySibling(); }

  /**
   * Get the heading sibling element for the peer
   * @public
   * @returns {HTMLElement|null}
   */
  getHeadingSibling() {
    return this._headingSibling;
  }

  get headingSibling() { return this.getHeadingSibling(); }

  /**
   * Get the label sibling element for the peer
   * @public
   * @returns {HTMLElement|null}
   */
  getLabelSibling() {
    return this._labelSibling;
  }

  get labelSibling() { return this.getLabelSibling(); }

  /**
   * Get the description sibling element for the peer
   * @public
   * @returns {HTMLElement|null}
   */
  getDescriptionSibling() {
    return this._descriptionSibling;
  }

  get descriptionSibling() { return this.getDescriptionSibling(); }

  /**
   * Get the container parent element for the peer
   * @public
   * @returns {HTMLElement|null}
   */
  getContainerParent() {
    return this._containerParent;
  }

  get containerParent() { return this.getContainerParent(); }

  /**
   * Returns the top-level element that contains the primary sibling. If there is no container parent, then the primary
   * sibling is returned.
   * @public
   *
   * @returns {HTMLElement|null}
   */
  getTopLevelElementContainingPrimarySibling() {
    return this._containerParent || this._primarySibling;
  }

  /**
   * Recompute the aria-labelledby attributes for all of the peer's elements
   * @public
   */
  onAriaLabelledbyAssociationChange() {
    this.removeAttributeFromAllElements( 'aria-labelledby' );

    for ( let i = 0; i < this.node.ariaLabelledbyAssociations.length; i++ ) {
      const associationObject = this.node.ariaLabelledbyAssociations[ i ];

      // Assert out if the model list is different than the data held in the associationObject
      assert && assert( associationObject.otherNode.nodesThatAreAriaLabelledbyThisNode.indexOf( this.node ) >= 0,
        'unexpected otherNode' );


      this.setAssociationAttribute( 'aria-labelledby', associationObject );
    }
  }

  /**
   * Recompute the aria-describedby attributes for all of the peer's elements
   * @public
   */
  onAriaDescribedbyAssociationChange() {
    this.removeAttributeFromAllElements( 'aria-describedby' );

    for ( let i = 0; i < this.node.ariaDescribedbyAssociations.length; i++ ) {
      const associationObject = this.node.ariaDescribedbyAssociations[ i ];

      // Assert out if the model list is different than the data held in the associationObject
      assert && assert( associationObject.otherNode.nodesThatAreAriaDescribedbyThisNode.indexOf( this.node ) >= 0,
        'unexpected otherNode' );


      this.setAssociationAttribute( 'aria-describedby', associationObject );
    }
  }

  /**
   * Recompute the aria-activedescendant attributes for all of the peer's elements
   * @public
   */
  onActiveDescendantAssociationChange() {
    this.removeAttributeFromAllElements( 'aria-activedescendant' );

    for ( let i = 0; i < this.node.activeDescendantAssociations.length; i++ ) {
      const associationObject = this.node.activeDescendantAssociations[ i ];

      // Assert out if the model list is different than the data held in the associationObject
      assert && assert( associationObject.otherNode.nodesThatAreActiveDescendantToThisNode.indexOf( this.node ) >= 0,
        'unexpected otherNode' );


      this.setAssociationAttribute( 'aria-activedescendant', associationObject );
    }
  }

  /**
   * Set the new attribute to the element if the value is a string. It will otherwise be null or undefined and should
   * then be removed from the element. This allows empty strings to be set as values.
   *
   * @param {string} key
   * @param {string|null|undefined} value
   * @private
   */
  handleAttributeWithPDOMOption( key, value ) {
    if ( typeof value === 'string' ) {
      this.setAttributeToElement( key, value );
    }
    else {
      this.removeAttributeFromElement( key );
    }
  }

  /**
   * Set all pdom attributes onto the peer elements from the model's stored data objects
   * @private
   *
   * @param {Object} [pdomOptions] - these can override the values of the node, see this.update()
   */
  onAttributeChange( pdomOptions ) {

    for ( let i = 0; i < this.node.pdomAttributes.length; i++ ) {
      const dataObject = this.node.pdomAttributes[ i ];
      this.setAttributeToElement( dataObject.attribute, dataObject.value, dataObject.options );
    }

    // Manually support options that map to attributes. This covers that case where behavior functions want to change
    // these, but they aren't in node.pdomAttributes. It will do double work in some cases, but it is pretty minor for
    // the complexity it saves. https://github.com/phetsims/scenery/issues/1436. Empty strings should be settable for
    // these attributes but null and undefined are ignored.
    this.handleAttributeWithPDOMOption( 'aria-label', pdomOptions.ariaLabel );
    this.handleAttributeWithPDOMOption( 'role', pdomOptions.ariaRole );
  }

  /**
   * Set all classes onto the peer elements from the model's stored data objects
   * @private
   */
  onClassChange() {
    for ( let i = 0; i < this.node.pdomClasses.length; i++ ) {
      const dataObject = this.node.pdomClasses[ i ];
      this.setClassToElement( dataObject.className, dataObject.options );
    }
  }

  /**
   * Set the input value on the peer's primary sibling element. The value attribute must be set as a Property to be
   * registered correctly by an assistive device. If null, the attribute is removed so that we don't clutter the DOM
   * with value="null" attributes.
   *
   * @public (scenery-internal)
   */
  onInputValueChange() {
    assert && assert( this.node.inputValue !== undefined, 'use null to remove input value attribute' );

    if ( this.node.inputValue === null ) {
      this.removeAttributeFromElement( 'value' );
    }
    else {

      // type conversion for DOM spec
      const valueString = `${this.node.inputValue}`;
      this.setAttributeToElement( 'value', valueString, { type: 'property' } );
    }
  }

  /**
   * Get an element on this node, looked up by the elementName flag passed in.
   * @public (scenery-internal)
   *
   * @param {string} elementName - see PDOMUtils for valid associations
   * @returns {HTMLElement}
   */
  getElementByName( elementName ) {
    if ( elementName === PDOMPeer.PRIMARY_SIBLING ) {
      return this._primarySibling;
    }
    else if ( elementName === PDOMPeer.LABEL_SIBLING ) {
      return this._labelSibling;
    }
    else if ( elementName === PDOMPeer.DESCRIPTION_SIBLING ) {
      return this._descriptionSibling;
    }
    else if ( elementName === PDOMPeer.CONTAINER_PARENT ) {
      return this._containerParent;
    }
    else if ( elementName === ACCESSIBLE_PARAGRAPH_SIBLING ) {
      return this._accessibleParagraphSibling;
    }
    else if ( elementName === HEADING_SIBLING ) {
      return this._headingSibling;
    }

    throw new Error( `invalid elementName name: ${elementName}` );
  }

  /**
   * Sets a attribute on one of the peer's window.Elements.
   * @public (scenery-internal)
   * @param {string} attribute
   * @param {*} attributeValue
   * @param {Object} [options]
   */
  setAttributeToElement( attribute, attributeValue, options ) {

    options = merge( {
      // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
      // for setting certain attributes (e.g. MathML).
      namespace: null,

      // set as a javascript property instead of an attribute on the DOM Element.
      type: 'attribute',

      elementName: PRIMARY_SIBLING, // see this.getElementName() for valid values, default to the primary sibling

      // {HTMLElement|null} - element that will directly receive the input rather than looking up by name, if
      // provided, elementName option will have no effect
      element: null
    }, options );

    // There may not be an element due to order of operations, or if there is no default primary sibling.
    const element = options.element || this.getElementByName( options.elementName );
    if ( !element ) {
      return;
    }

    // For dynamic strings, we may need to retrieve the actual value.
    const rawAttributeValue = PDOMUtils.unwrapProperty( attributeValue );

    // remove directional formatting that may surround strings if they are translatable
    let attributeValueWithoutMarks = rawAttributeValue;
    if ( typeof rawAttributeValue === 'string' ) {
      attributeValueWithoutMarks = stripEmbeddingMarks( rawAttributeValue );
    }

    if ( attribute === DISABLED_ATTRIBUTE_NAME && !this.display.interactive ) {

      // The presence of the `disabled` attribute means it is always disabled.
      this._preservedDisabledValue = options.type === 'property' ? attributeValueWithoutMarks : true;
    }

    if ( options.namespace ) {
      element.setAttributeNS( options.namespace, attribute, attributeValueWithoutMarks );
    }
    else if ( options.type === 'property' ) {
      element[ attribute ] = attributeValueWithoutMarks;
    }
    else {
      element.setAttribute( attribute, attributeValueWithoutMarks );
    }
  }

  /**
   * Remove attribute from one of the peer's window.Elements.
   * @public (scenery-internal)
   * @param {string} attribute
   * @param {Object} [options]
   */
  removeAttributeFromElement( attribute, options ) {

    options = merge( {
      // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
      // for setting certain attributes (e.g. MathML).
      namespace: null,

      elementName: PRIMARY_SIBLING, // see this.getElementName() for valid values, default to the primary sibling

      // {HTMLElement|null} - element that will directly receive the input rather than looking up by name, if
      // provided, elementName option will have no effect
      element: null
    }, options );

    const element = options.element || this.getElementByName( options.elementName );

    if ( options.namespace ) {
      element.removeAttributeNS( options.namespace, attribute );
    }
    else if ( attribute === DISABLED_ATTRIBUTE_NAME && !this.display.interactive ) {
      // maintain our interal disabled state in case the display toggles back to be interactive.
      this._preservedDisabledValue = false;
    }
    else {
      element.removeAttribute( attribute );
    }
  }

  /**
   * Remove the given attribute from all peer elements
   * @public (scenery-internal)
   * @param {string} attribute
   */
  removeAttributeFromAllElements( attribute ) {
    assert && assert( attribute !== DISABLED_ATTRIBUTE_NAME, 'this method does not currently support disabled, to make Display.interactive toggling easier to implement' );
    assert && assert( typeof attribute === 'string' );
    this._primarySibling && this._primarySibling.removeAttribute( attribute );
    this._labelSibling && this._labelSibling.removeAttribute( attribute );
    this._descriptionSibling && this._descriptionSibling.removeAttribute( attribute );
    this._accessibleParagraphSibling && this._accessibleParagraphSibling.removeAttribute( attribute );
    this._containerParent && this._containerParent.removeAttribute( attribute );
  }

  /**
   * Add the provided className to the element's classList.
   *
   * @public
   * @param {string} className
   * @param {Object} [options]
   */
  setClassToElement( className, options ) {
    assert && assert( typeof className === 'string' );

    options = merge( {

      // Name of the element who we are adding the class to, see this.getElementName() for valid values
      elementName: PRIMARY_SIBLING
    }, options );

    this.getElementByName( options.elementName ).classList.add( className );
  }

  /**
   * Remove the specified className from the element.
   * @public
   *
   * @param {string} className
   * @param {Object} [options]
   */
  removeClassFromElement( className, options ) {
    assert && assert( typeof className === 'string' );

    options = merge( {

      // Name of the element who we are removing the class from, see this.getElementName() for valid values
      elementName: PRIMARY_SIBLING
    }, options );

    this.getElementByName( options.elementName ).classList.remove( className );
  }

  /**
   * Set either association attribute (aria-labelledby/describedby) on one of this peer's Elements
   * @public (scenery-internal)
   * @param {string} attribute - either aria-labelledby or aria-describedby
   * @param {Object} associationObject - see addAriaLabelledbyAssociation() for schema
   */
  setAssociationAttribute( attribute, associationObject ) {
    assert && assert( PDOMUtils.ASSOCIATION_ATTRIBUTES.indexOf( attribute ) >= 0,
      `unsupported attribute for setting with association object: ${attribute}` );

    const otherNodePDOMInstances = associationObject.otherNode.getPDOMInstances();

    // If the other node hasn't been added to the scene graph yet, it won't have any accessible instances, so no op.
    // This will be recalculated when that node is added to the scene graph
    if ( otherNodePDOMInstances.length > 0 ) {

      // We are just using the first PDOMInstance for simplicity, but it is OK because the accessible
      // content for all PDOMInstances will be the same, so the Accessible Names (in the browser's
      // accessibility tree) of elements that are referenced by the attribute value id will all have the same content
      const firstPDOMInstance = otherNodePDOMInstances[ 0 ];

      // Handle a case where you are associating to yourself, and the peer has not been constructed yet.
      if ( firstPDOMInstance === this.pdomInstance ) {
        firstPDOMInstance.peer = this;
      }

      assert && assert( firstPDOMInstance.peer, 'peer should exist' );

      // we can use the same element's id to update all of this Node's peers
      const otherPeerElement = firstPDOMInstance.peer.getElementByName( associationObject.otherElementName );

      const element = this.getElementByName( associationObject.thisElementName );

      // to support any option order, no-op if the peer element has not been created yet.
      if ( element && otherPeerElement ) {

        // only update associations if the requested peer element has been created
        // NOTE: in the future, we would like to verify that the association exists but can't do that yet because
        // we have to support cases where we set label association prior to setting the sibling/parent tagName
        const previousAttributeValue = element.getAttribute( attribute ) || '';
        assert && assert( typeof previousAttributeValue === 'string' );

        const newAttributeValue = [ previousAttributeValue.trim(), otherPeerElement.id ].join( ' ' ).trim();

        // add the id from the new association to the value of the HTMLElement's attribute.
        this.setAttributeToElement( attribute, newAttributeValue, {
          elementName: associationObject.thisElementName
        } );
      }
    }
  }

  /**
   * The contentElement will either be a label or description element. The contentElement will be sorted relative to
   * the primarySibling. Its placement will also depend on whether or not this node wants to append this element,
   * see setAppendLabel() and setAppendDescription(). By default, the "content" element will be placed before the
   * primarySibling.
   *
   * NOTE: This function assumes it is called on label sibling before description sibling for inserting elements
   * into the correct order.
   *
   * @private
   *
   * @param {HTMLElement} contentElement
   * @param {boolean} appendElement
   */
  arrangeContentElement( contentElement, appendElement ) {

    // if there is a containerParent
    if ( this.topLevelElements[ 0 ] === this._containerParent ) {
      assert && assert( this.topLevelElements.length === 1 );

      if ( appendElement ) {
        this._containerParent.appendChild( contentElement );
      }
      else {
        this._containerParent.insertBefore( contentElement, this._primarySibling );
      }
    }

    // If there are multiple top level nodes
    else {
      if ( assert && !appendElement ) {
        assert && assert( this._primarySibling, 'There must be a primary sibling to sort relative to it if not appending.' );
      }

      // keep this.topLevelElements in sync
      arrayRemove( this.topLevelElements, contentElement );
      const indexOfPrimarySibling = this.topLevelElements.indexOf( this._primarySibling );

      // if appending, just insert at at end of the top level elements
      const insertIndex = appendElement ? this.topLevelElements.length : indexOfPrimarySibling;
      this.topLevelElements.splice( insertIndex, 0, contentElement );
    }
  }

  /**
   * Is this peer hidden in the PDOM
   * @public
   *
   * @returns {boolean}
   */
  isVisible() {
    if ( assert ) {

      let visibleElements = 0;
      this.topLevelElements.forEach( element => {

        // support property or attribute
        if ( !element.hidden && !element.hasAttribute( 'hidden' ) ) {
          visibleElements += 1;
        }
      } );
      assert( this.visible ? visibleElements === this.topLevelElements.length : visibleElements === 0,
        'some of the peer\'s elements are visible and some are not' );

    }
    return this.visible === null ? true : this.visible; // default to true if visibility hasn't been set yet.
  }

  /**
   * Set whether or not the peer is visible in the PDOM
   * @public
   *
   * @param {boolean} visible
   */
  setVisible( visible ) {
    assert && assert( typeof visible === 'boolean' );
    if ( this.visible !== visible ) {

      this.visible = visible;
      for ( let i = 0; i < this.topLevelElements.length; i++ ) {
        const element = this.topLevelElements[ i ];
        if ( visible ) {
          this.removeAttributeFromElement( 'hidden', { element: element } );
        }
        else {
          this.setAttributeToElement( 'hidden', '', { element: element } );
        }
      }

      // Invalidate CSS transforms because when 'hidden' the content will have no dimensions in the viewport. For
      // a Safari VoiceOver bug, also force a reflow in the next animation frame to ensure that the accessible name is
      // correct.
      // TODO: Remove this when the bug is fixed. See https://github.com/phetsims/a11y-research/issues/193
      this.invalidateCSSPositioning( platform.safari );
    }
  }

  /**
   * Returns if this peer is focused. A peer is focused if its primarySibling is focused.
   * @public (scenery-internal)
   * @returns {boolean}
   */
  isFocused() {
    const visualFocusTrail = guessVisualTrail( this.trail, this.display.rootNode );

    return pdomFocusProperty.value && pdomFocusProperty.value.trail.equals( visualFocusTrail );
  }

  /**
   * Focus the primary sibling of the peer. If this peer is not visible, this is a no-op (native behavior).
   * @public (scenery-internal)
   */
  focus() {
    assert && assert( this._primarySibling, 'must have a primary sibling to focus' );

    // We do not want to steal focus from any parent application. For example, if this element is in an iframe.
    // See https://github.com/phetsims/joist/issues/897.
    if ( FocusManager.windowHasFocusProperty.value ) {
      this._primarySibling.focus();
    }
  }

  /**
   * Blur the primary sibling of the peer.
   * @public (scenery-internal)
   */
  blur() {
    assert && assert( this._primarySibling, 'must have a primary sibling to blur' );

    // no op by the browser if primary sibling does not have focus
    this._primarySibling.blur();
  }

  /**
   * Make the peer focusable. Only the primary sibling is ever considered focusable.
   * @public
   * @param {boolean} focusable
   */
  setFocusable( focusable ) {
    assert && assert( typeof focusable === 'boolean' );

    const peerHadFocus = this.isFocused();
    if ( this.focusable !== focusable && this.primarySibling ) {
      this.focusable = focusable;
      PDOMUtils.overrideFocusWithTabIndex( this.primarySibling, focusable );

      // in Chrome, if tabindex is removed and the element is not focusable by default the element is blurred.
      // This behavior is reasonable and we want to enforce it in other browsers for consistency. See
      // https://github.com/phetsims/scenery/issues/967
      if ( peerHadFocus && !focusable ) {
        this.blur();
      }

      // reposition the sibling in the DOM, since non-focusable nodes are not positioned
      this.invalidateCSSPositioning();
    }
  }

  /**
   * Sets the heading content
   * @public (scenery-internal)
   * @param {string|null} content --- NOTE that this is called from update(), where the value gets string-ified, so this
   * type info is correct
   */
  setHeadingContent( content ) {
    assert && assert( content === null || typeof content === 'string', 'incorrect heading content type' );

    // no-op to support any option order
    if ( !this._headingSibling ) {
      return;
    }

    PDOMUtils.setTextContent( this._headingSibling, content );
  }

  /**
   * Responsible for setting the content for the label sibling
   * @public (scenery-internal)
   * @param {string|null} content - the content for the label sibling.
   */
  setLabelSiblingContent( content ) {
    assert && assert( content === null || typeof content === 'string', 'incorrect label content type' );

    // no-op to support any option order
    if ( !this._labelSibling ) {
      return;
    }

    PDOMUtils.setTextContent( this._labelSibling, content );
  }

  /**
   * Responsible for setting the content for the description sibling
   * @public (scenery-internal)
   * @param {string|null} content - the content for the description sibling.
   */
  setDescriptionSiblingContent( content ) {
    assert && assert( content === null || typeof content === 'string', 'incorrect description content type' );

    // no-op to support any option order
    if ( !this._descriptionSibling ) {
      return;
    }
    PDOMUtils.setTextContent( this._descriptionSibling, content );
  }

  /**
   * Responsible for setting the content for the paragraph sibling.
   * @public (scenery-internal)
   * @param {string|null} content - the content for the paragraph sibling.
   */
  setAccessibleParagraphContent( content ) {
    assert && assert( content === null || typeof content === 'string', 'incorrect description content type' );

    // no-op to support any option order
    if ( !this._accessibleParagraphSibling ) {
      return;
    }
    PDOMUtils.setTextContent( this._accessibleParagraphSibling, content );
  }

  /**
   * Responsible for setting the content for the primary sibling
   * @public (scenery-internal)
   * @param {string|null} content - the content for the primary sibling.
   */
  setPrimarySiblingContent( content ) {

    // no-op to support any option order
    if ( !this._primarySibling ) {
      return;
    }

    assert && assert( content === null || typeof content === 'string', 'incorrect inner content type' );
    assert && assert( this.pdomInstance.children.length === 0, 'descendants exist with accessible content, innerContent cannot be used' );
    assert && assert( PDOMUtils.tagNameSupportsContent( this._primarySibling.tagName ),
      `tagName: ${this.node.tagName} does not support inner content` );

    PDOMUtils.setTextContent( this._primarySibling, content );
  }

  /**
   * Sets the pdomTransformSourceNode so that the primary sibling will be transformed with changes to along the
   * unique trail to the source node. If null, repositioning happens with transform changes along this
   * pdomInstance's trail.
   * @public
   *
   * @param {../nodes/Node|null} node
   */
  setPDOMTransformSourceNode( node ) {

    // remove previous listeners before creating a new TransformTracker
    this.pdomInstance.transformTracker.removeListener( this.transformListener );
    this.pdomInstance.updateTransformTracker( node );

    // add listeners back after update
    this.pdomInstance.transformTracker.addListener( this.transformListener );

    // new trail with transforms so positioning is probably dirty
    this.invalidateCSSPositioning();
  }

  /**
   * Enable or disable positioning of the sibling elements. Generally this is requiredfor accessibility to work on
   * touch screen based screen readers like phones. But repositioning DOM elements is expensive. This can be set to
   * false to optimize when positioning is not necessary.
   * @public (scenery-internal)
   *
   * @param {boolean} positionInPDOM
   */
  setPositionInPDOM( positionInPDOM ) {
    this.positionInPDOM = positionInPDOM;

    // signify that it needs to be repositioned next frame, either off screen or to match
    // graphical rendering
    this.invalidateCSSPositioning();
  }

  // @private
  getElementId( siblingName, stringId ) {
    return `display${this.display.id}-${siblingName}-${stringId}`;
  }


  /**
   * Set ids on elements so for easy lookup with document.getElementById. Also assign a unique
   * data attribute to the elements so that scenery can look up an element from a Trail (mostly
   * for input handling).
   *
   * Note that dataset isn't supported by all namespaces (like MathML) so we need to use setAttribute.
   * @public
   */
  updateIndicesStringAndElementIds() {
    const indices = this.pdomInstance.getPDOMInstanceUniqueId();

    if ( this._primarySibling ) {
      this._primarySibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._primarySibling.id = this.getElementId( 'primary', indices );
    }
    if ( this._labelSibling ) {
      this._labelSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._labelSibling.id = this.getElementId( 'label', indices );
    }
    if ( this._descriptionSibling ) {
      this._descriptionSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._descriptionSibling.id = this.getElementId( 'description', indices );
    }
    if ( this._headingSibling ) {
      this._headingSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._headingSibling.id = this.getElementId( 'heading', indices );
    }
    if ( this._accessibleParagraphSibling ) {
      this._accessibleParagraphSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._accessibleParagraphSibling.id = this.getElementId( 'paragraph', indices );
    }
    if ( this._containerParent ) {
      this._containerParent.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._containerParent.id = this.getElementId( 'container', indices );
    }
  }

  /**
   * Mark that the siblings of this PDOMPeer need to be updated in the next Display update. Possibly from a
   * change of accessible content or node transformation. Does nothing if already marked dirty.
   *
   * @param [forceReflowWorkaround] - In addition to repositioning, force a reflow next animation frame? See
   *                                  this.forceReflowWorkaround for more information.
   * @private
   */
  invalidateCSSPositioning( forceReflowWorkaround = false ) {
    if ( !this.positionDirty ) {
      this.positionDirty = true;

      if ( forceReflowWorkaround ) {
        this.forceReflowWorkaround = true;

        // `transform=scale(1)` forces a reflow so we can set this and revert it in the next animation frame.
        // Transform is used instead of `display='none'` because changing display impacts focus.
        for ( let i = 0; i < this.topLevelElements.length; i++ ) {
          this.topLevelElements[ i ].style.transform = 'scale(1)';
        }
      }

      // mark all ancestors of this peer so that we can quickly find this dirty peer when we traverse
      // the PDOMInstance tree
      let parent = this.pdomInstance.parent;
      while ( parent ) {
        parent.peer.childPositionDirty = true;
        parent = parent.parent;
      }
    }
  }

  /**
   * Update the CSS positioning of the primary and label siblings. Required to support accessibility on mobile
   * devices. On activation of focusable elements, certain AT will send fake pointer events to the browser at
   * the center of the client bounding rectangle of the HTML element. By positioning elements over graphical display
   * objects we can capture those events. A transformation matrix is calculated that will transform the position
   * and dimension of the HTML element in pixels to the global coordinate frame. The matrix is used to transform
   * the bounds of the element prior to any other transformation so we can set the element's left, top, width, and
   * height with CSS attributes.
   *
   * For now we are only transforming the primary and label siblings if the primary sibling is focusable. If
   * focusable, the primary sibling needs to be transformed to receive user input. VoiceOver includes the label bounds
   * in its calculation for where to send the events, so it needs to be transformed as well. Descriptions are not
   * considered and do not need to be positioned.
   *
   * Initially, we tried to set the CSS transformations on elements directly through the transform attribute. While
   * this worked for basic input, it did not support other AT features like tapping the screen to focus elements.
   * With this strategy, the VoiceOver "touch area" was a small box around the top left corner of the element. It was
   * never clear why this was this case, but forced us to change our strategy to set the left, top, width, and height
   * attributes instead.
   *
   * This function assumes that elements have other style attributes so they can be positioned correctly and don't
   * interfere with scenery input, see SceneryStyle in PDOMUtils.
   *
   * Additional notes were taken in https://github.com/phetsims/scenery/issues/852, see that issue for more
   * information.
   *
   * Review: This function could be simplified by setting the element width/height a small arbitrary shape
   * at the center of the node's global bounds. There is a drawback in that the VO default highlight won't
   * surround the Node anymore. But it could be a performance enhancement and simplify this function.
   * Or maybe a big rectangle larger than the Display div still centered on the node so we never
   * see the VO highlight?
   *
   * @private
   */
  positionElements( positionInPDOM ) {
    const placeableSibling = this.getPlaceableSibling();
    assert && assert( placeableSibling, 'a primary sibling required to receive CSS positioning' );
    assert && assert( this.positionDirty, 'elements should only be repositioned if dirty' );

    // CSS transformation only needs to be applied if the node is focusable - otherwise the element will be found
    // by gesture navigation with the virtual cursor. Bounds for non-focusable elements in the ViewPort don't
    // need to be accurate because the AT doesn't need to send events to them.
    if ( positionInPDOM ) {
      const transformSourceNode = this.node.pdomTransformSourceNode || this.node;

      scratchGlobalBounds.set( transformSourceNode.localBounds );
      if ( scratchGlobalBounds.isFinite() ) {
        scratchGlobalBounds.transform( this.pdomInstance.transformTracker.getMatrix() );

        // no need to position if the node is fully outside of the Display bounds (out of view)
        const displayBounds = this.display.bounds;
        if ( displayBounds.intersectsBounds( scratchGlobalBounds ) ) {

          // Constrain the global bounds to Display bounds so that center of the sibling element
          // is always in the Display. We may miss input if the center of the Node is outside
          // the Display, where VoiceOver would otherwise send pointer events.
          scratchGlobalBounds.constrainBounds( displayBounds );

          let clientDimensions = getClientDimensions( placeableSibling );
          let clientWidth = clientDimensions.width;
          let clientHeight = clientDimensions.height;

          if ( clientWidth > 0 && clientHeight > 0 ) {
            scratchSiblingBounds.setMinMax( 0, 0, clientWidth, clientHeight );
            scratchSiblingBounds.transform( getCSSMatrix( clientWidth, clientHeight, scratchGlobalBounds ) );
            setClientBounds( placeableSibling, scratchSiblingBounds );
          }

          if ( this.labelSibling ) {
            clientDimensions = getClientDimensions( this._labelSibling );
            clientWidth = clientDimensions.width;
            clientHeight = clientDimensions.height;

            if ( clientHeight > 0 && clientWidth > 0 ) {
              scratchSiblingBounds.setMinMax( 0, 0, clientWidth, clientHeight );
              scratchSiblingBounds.transform( getCSSMatrix( clientWidth, clientHeight, scratchGlobalBounds ) );
              setClientBounds( this._labelSibling, scratchSiblingBounds );
            }
          }
        }
      }
    }
    else {

      // not positioning, just move off screen
      scratchSiblingBounds.set( PDOMPeer.OFFSCREEN_SIBLING_BOUNDS );
      setClientBounds( placeableSibling, scratchSiblingBounds );
      if ( this._labelSibling ) {
        setClientBounds( this._labelSibling, scratchSiblingBounds );
      }
    }

    if ( this.forceReflowWorkaround ) {

      // Force a reflow (recalculation of DOM layout) to fix the accessible name.
      this.topLevelElements.forEach( element => {
        element.style.transform = ''; // force reflow request by removing the transform added in the previous frame
        element.style.offsetHeight; // query the offsetHeight after restoring display to force reflow
      } );
    }

    this.positionDirty = false;
    this.forceReflowWorkaround = false;
  }

  /**
   * Update positioning of elements in the PDOM. Does a depth first search for all descendants of parentIntsance with
   * a peer that either has dirty positioning or as a descendant with dirty positioning.
   *
   * @public (scenery-internal)
   */
  updateSubtreePositioning( parentPositionInPDOM = false ) {
    this.childPositionDirty = false;

    const positionInPDOM = this.positionInPDOM || parentPositionInPDOM;

    if ( this.positionDirty ) {
      this.positionElements( positionInPDOM );
    }

    for ( let i = 0; i < this.pdomInstance.children.length; i++ ) {
      const childPeer = this.pdomInstance.children[ i ].peer;
      if ( childPeer.positionDirty || childPeer.childPositionDirty ) {
        this.pdomInstance.children[ i ].peer.updateSubtreePositioning( positionInPDOM );
      }
    }
  }

  /**
   * Recursively set this PDOMPeer and children to be disabled. This will overwrite any previous value of disabled
   * that may have been set, but will keep track of the old value, and restore its state upon re-enabling.
   * @param {boolean} disabled
   * @public
   */
  recursiveDisable( disabled ) {

    if ( this._primarySibling ) {
      if ( disabled ) {
        this._preservedDisabledValue = this._primarySibling.disabled;
        this._primarySibling.disabled = true;
      }
      else {
        this._primarySibling.disabled = this._preservedDisabledValue;
      }
    }

    for ( let i = 0; i < this.pdomInstance.children.length; i++ ) {
      this.pdomInstance.children[ i ].peer.recursiveDisable( disabled );
    }
  }

  /**
   * Removes external references from this peer, and places it in the pool.
   * @public (scenery-internal)
   */
  dispose() {
    this.isDisposed = true;

    // remove focus if the disposed peer is the active element
    if ( this._primarySibling ) {
      this.blur();

      this._primarySibling.removeEventListener( 'blur', this.blurEventListener );
      this._primarySibling.removeEventListener( 'focus', this.focusEventListener );
    }

    // remove listeners
    this.pdomInstance.transformTracker.removeListener( this.transformListener );
    this.mutationObserver.disconnect();

    // zero-out references
    this.pdomInstance = null;
    this.node = null;
    this.display = null;
    this.trail = null;
    this._primarySibling = null;
    this._labelSibling = null;
    this._descriptionSibling = null;
    this._headingSibling = null;
    this._accessibleParagraphSibling = null;
    this._containerParent = null;
    this.focusable = null;

    // for now
    this.freeToPool();
  }
}

// @public {string} - specifies valid associations between related PDOMPeers in the DOM
PDOMPeer.PRIMARY_SIBLING = PRIMARY_SIBLING; // associate with all accessible content related to this peer
PDOMPeer.HEADING_SIBLING = HEADING_SIBLING; // associate with just the heading content of this peer
PDOMPeer.LABEL_SIBLING = LABEL_SIBLING; // associate with just the label content of this peer
PDOMPeer.DESCRIPTION_SIBLING = DESCRIPTION_SIBLING; // associate with just the description content of this peer
PDOMPeer.PARAGRAPH_SIBLING = ACCESSIBLE_PARAGRAPH_SIBLING; // associate with just the paragraph content of this peer
PDOMPeer.CONTAINER_PARENT = CONTAINER_PARENT; // associate with everything under the container parent of this peer

// @public (scenery-internal) - bounds for a sibling that should be moved off-screen when not positioning, in
// global coordinates
PDOMPeer.OFFSCREEN_SIBLING_BOUNDS = new Bounds2( 0, 0, 1, 1 );

scenery.register( 'PDOMPeer', PDOMPeer );

// Set up pooling
Poolable.mixInto( PDOMPeer, {
  initialize: PDOMPeer.prototype.initializePDOMPeer
} );

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------

/**
 * Create a sibling element for the PDOMPeer.
 * TODO: this should be inlined with the PDOMUtils method https://github.com/phetsims/scenery/issues/1581
 * @param {string} tagName
 * @param {boolean} focusable
 * @param {Object} [options] - passed along to PDOMUtils.createElement
 * @returns {HTMLElement}
 */
function createElement( tagName, focusable, options ) {
  options = merge( {

    // {string|null} - addition to the trailId, separated by a hyphen to identify the different siblings within
    // the document
    siblingName: null,

    // {boolean} - if true, DOM input events received on the element will not be dispatched as SceneryEvents in Input.js
    // see ParallelDOM.setExcludeLabelSiblingFromInput for more information
    excludeFromInput: false
  }, options );

  const newElement = PDOMUtils.createElement( tagName, focusable, options );

  if ( options.excludeFromInput ) {
    newElement.setAttribute( PDOMUtils.DATA_EXCLUDE_FROM_INPUT, true );
  }

  return newElement;
}

/**
 * Get a matrix that can be used as the CSS transform for elements in the DOM. This matrix will an HTML element
 * dimensions in pixels to the global coordinate frame.
 *
 * @param  {number} clientWidth - width of the element to transform in pixels
 * @param  {number} clientHeight - height of the element to transform in pixels
 * @param  {Bounds2} nodeGlobalBounds - Bounds of the PDOMPeer's node in the global coordinate frame.
 * @returns {Matrix3}
 */
function getCSSMatrix( clientWidth, clientHeight, nodeGlobalBounds ) {

  // the translation matrix for the node's bounds in its local coordinate frame
  globalNodeTranslationMatrix.setToTranslation( nodeGlobalBounds.minX, nodeGlobalBounds.minY );

  // scale matrix for "client" HTML element, scale to make the HTML element's DOM bounds match the
  // local bounds of the node
  globalToClientScaleMatrix.setToScale( nodeGlobalBounds.width / clientWidth, nodeGlobalBounds.height / clientHeight );

  // combine these in a single transformation matrix
  return globalNodeTranslationMatrix.multiplyMatrix( globalToClientScaleMatrix ).multiplyMatrix( nodeScaleMagnitudeMatrix );
}

/**
 * Gets an object with the width and height of an HTML element in pixels, prior to any scaling. clientWidth and
 * clientHeight are zero for elements with inline layout and elements without CSS. For those elements we fall back
 * to the boundingClientRect, which at that point will describe the dimensions of the element prior to scaling.
 *
 * @param  {HTMLElement} siblingElement
 * @returns {Object} - Returns an object with two entries, { width: {number}, height: {number} }
 */
function getClientDimensions( siblingElement ) {
  let clientWidth = siblingElement.clientWidth;
  let clientHeight = siblingElement.clientHeight;

  if ( clientWidth === 0 && clientHeight === 0 ) {
    const clientRect = siblingElement.getBoundingClientRect();
    clientWidth = clientRect.width;
    clientHeight = clientRect.height;
  }

  return { width: clientWidth, height: clientHeight };
}

/**
 * Set the bounds of the sibling element in the view port in pixels, using top, left, width, and height css.
 * The element must be styled with 'position: fixed', and an ancestor must have position: 'relative', so that
 * the dimensions of the sibling are relative to the parent.
 *
 * @param {HTMLElement} siblingElement - the element to position
 * @param {Bounds2} bounds - desired bounds, in pixels
 */
function setClientBounds( siblingElement, bounds ) {
  siblingElement.style.top = `${bounds.top}px`;
  siblingElement.style.left = `${bounds.left}px`;
  siblingElement.style.width = `${bounds.width}px`;
  siblingElement.style.height = `${bounds.height}px`;
}

export default PDOMPeer;